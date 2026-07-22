"""Shared DrugOOD IC50 data, GIN, training, and evaluation utilities."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Sampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool


def discover_data_root() -> Path:
    """Find the DrugOOD cache on both the host and `/workspace` container layout."""
    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "Graph-OOD-Lab" / "data" / "DrugOOD",
        project_root / "CIGA" / "data" / "DrugOOD",
        Path("/workspace/Graph-OOD-Lab/data/DrugOOD"),
        Path("/workspace/CIGA/data/DrugOOD"),
        Path("/home/jylim/project/Graph-OOD-Lab/data/DrugOOD"),
        Path("/home/jylim/project/CIGA/data/DrugOOD"),
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


DEFAULT_DATA_ROOT = discover_data_root()


class CachedDrugOOD(InMemoryDataset):
    """Read the `(data, slices)` files produced by the existing DrugOOD converter."""

    def __init__(self, path: Path):
        super().__init__(root=None)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing preprocessed split: {path}\n"
                "Generate the corresponding DrugOOD PyG cache first. The currently "
                "available caches are under Graph-OOD-Lab/data/DrugOOD."
            )
        self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)


class DrugOODGIN(nn.Module):
    """Four-layer GIN used consistently across the three baselines."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch) -> Tensor:
        x = self.node_encoder(batch.x.float())
        edge_attr = batch.edge_attr.float()
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, batch.edge_index, edge_attr))
            if layer + 1 < len(self.convs):
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_repr = global_add_pool(x, batch.batch)
        return self.classifier(graph_repr)


class GroupBatchSampler(Sampler[list[int]]):
    """WILDS/DrugOOD-style batches with uniformly sampled distinct groups.

    Every batch first samples ``n_groups_per_batch`` groups uniformly and then
    samples an approximately equal number of examples from each selected group.
    Sampling examples with replacement keeps small groups from disappearing.
    """

    def __init__(
        self,
        dataset: CachedDrugOOD,
        batch_size: int,
        n_groups_per_batch: int,
        seed: int,
    ) -> None:
        if batch_size < n_groups_per_batch:
            raise ValueError("batch_size must be at least n_groups_per_batch")
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            group = int(dataset[index].group.view(-1)[0])
            grouped[group].append(index)
        if len(grouped) < n_groups_per_batch:
            raise ValueError(
                f"Requested {n_groups_per_batch} groups per batch, but dataset has {len(grouped)}"
            )
        self.group_indices = grouped
        self.groups = tuple(sorted(grouped))
        self.batch_size = batch_size
        self.n_groups_per_batch = n_groups_per_batch
        self.num_batches = math.ceil(len(dataset) / batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        base, remainder = divmod(self.batch_size, self.n_groups_per_batch)
        for _ in range(self.num_batches):
            selected = generator.sample(self.groups, self.n_groups_per_batch)
            batch = []
            for position, group in enumerate(selected):
                count = base + int(position < remainder)
                indices = self.group_indices[group]
                batch.extend(generator.choices(indices, k=count))
            generator.shuffle(batch)
            yield batch


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--domain", choices=("assay", "scaffold", "size"), default="assay")
    parser.add_argument("--subset", choices=("core", "general", "refined"), default="core")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset stem override, e.g. drugood_lbap_core_ic50_assay.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience based on OOD validation performance.",
    )
    parser.add_argument(
        "--erm-pretrain-epochs",
        type=int,
        default=10,
        help="ERM pretraining epochs for non-ERM DrugOOD methods.",
    )
    parser.add_argument(
        "--groups-per-batch",
        type=int,
        default=4,
        help="Number of distinct groups per IRM/V-REx/GroupDRO batch.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({name}) but is not available")
    return device


def dataset_stem(args: argparse.Namespace) -> str:
    return args.dataset or f"drugood_lbap_{args.subset}_ic50_{args.domain}"


def load_splits(args: argparse.Namespace) -> Dict[str, CachedDrugOOD]:
    stem = dataset_stem(args)
    required = ("train", "ood_val", "ood_test")
    optional = ("iid_val", "iid_test")
    splits: Dict[str, CachedDrugOOD] = {}
    for split in required:
        splits[split] = CachedDrugOOD(args.data_root / f"{stem}_{split}.pt")
    for split in optional:
        path = args.data_root / f"{stem}_{split}.pt"
        if path.is_file():
            splits[split] = CachedDrugOOD(path)
    return splits


def group_risks(losses: Tensor, groups: Tensor) -> tuple[Tensor, Tensor]:
    unique_groups = torch.unique(groups, sorted=True)
    risks = torch.stack([losses[groups == group].mean() for group in unique_groups])
    return risks, unique_groups


def irm_penalty(logits: Tensor, labels: Tensor, groups: Tensor) -> Tensor:
    penalties = []
    for group in torch.unique(groups):
        mask = groups == group
        scale = torch.ones((), device=logits.device, requires_grad=True)
        risk = F.cross_entropy(logits[mask] * scale, labels[mask])
        grad = torch.autograd.grad(risk, scale, create_graph=True)[0]
        penalties.append(grad.square())
    return torch.stack(penalties).mean() if penalties else logits.sum() * 0.0


def compute_objective(
    algorithm: str,
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    args: argparse.Namespace,
    state: dict,
) -> tuple[Tensor, dict]:
    losses = F.cross_entropy(logits, labels, reduction="none")
    risks, present_groups = group_risks(losses, groups)
    details = {"erm": float(losses.mean().detach())}

    if algorithm == "erm":
        return losses.mean(), details

    if algorithm == "irm":
        penalty = irm_penalty(logits, labels, groups)
        penalty_weight = (
            args.penalty_weight
            if state["update_count"] >= args.penalty_anneal_steps
            else 1.0
        )
        details["penalty"] = float(penalty.detach())
        details["penalty_weight"] = float(penalty_weight)
        return risks.mean() + penalty_weight * penalty, details

    if algorithm == "vrex":
        penalty = risks.var(unbiased=False)
        penalty_weight = (
            args.penalty_weight
            if state["update_count"] >= args.penalty_anneal_steps
            else 1.0
        )
        details["penalty"] = float(penalty.detach())
        details["penalty_weight"] = float(penalty_weight)
        return risks.mean() + penalty_weight * penalty, details

    if algorithm == "groupdro":
        group_values = state["group_values"]
        q = state["group_weights"]
        positions = torch.searchsorted(group_values, present_groups)
        full_risks = losses.new_zeros(group_values.numel())
        full_risks[positions] = risks
        with torch.no_grad():
            q *= torch.exp((args.step_size * full_risks.detach()).clamp(max=50))
            q /= q.sum().clamp_min(1e-12)
        objective = torch.dot(full_risks, q)
        details["worst_group_risk"] = float(risks.max().detach())
        return objective, details

    raise ValueError(f"Unknown algorithm: {algorithm}")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return math.nan
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_scores, all_preds = [], [], []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        logits = model(batch)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total_count += labels.numel()
        all_labels.append(labels.cpu())
        all_scores.append(logits.softmax(dim=-1)[:, 1].cpu())
        all_preds.append(logits.argmax(dim=-1).cpu())
    labels_np = torch.cat(all_labels).numpy()
    scores_np = torch.cat(all_scores).numpy()
    preds_np = torch.cat(all_preds).numpy()
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": float((preds_np == labels_np).mean()),
        "roc_auc": binary_auc(labels_np, scores_np),
        "count": int(total_count),
    }


def make_loader(
    dataset,
    args: argparse.Namespace,
    shuffle: bool,
    group_balanced: bool = False,
) -> DataLoader:
    common = dict(
        dataset=dataset,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if group_balanced:
        batch_sampler = GroupBatchSampler(
            dataset,
            batch_size=args.batch_size,
            n_groups_per_batch=args.groups_per_batch,
            seed=args.seed,
        )
        return DataLoader(batch_sampler=batch_sampler, **common)
    return DataLoader(batch_size=args.batch_size, shuffle=shuffle, **common)


def infer_feature_dims(dataset: CachedDrugOOD) -> tuple[int, int]:
    sample = dataset[0]
    node_dim = 1 if sample.x.ndim == 1 else sample.x.shape[-1]
    edge_dim = 1 if sample.edge_attr.ndim == 1 else sample.edge_attr.shape[-1]
    return int(node_dim), int(edge_dim)


def all_group_values(dataset: CachedDrugOOD) -> Tensor:
    values = sorted({int(dataset[index].group.view(-1)[0]) for index in range(len(dataset))})
    return torch.tensor(values, dtype=torch.long)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    algorithm: str,
    args: argparse.Namespace,
    state: dict,
) -> tuple[float, dict, torch.optim.Optimizer]:
    model.train()
    running_loss = 0.0
    seen = 0
    last_details = {}
    for batch in loader:
        batch = batch.to(device)
        labels = batch.y.view(-1).long()
        groups = batch.group.view(-1).long()
        if (
            algorithm in {"irm", "vrex"}
            and state["update_count"] == args.penalty_anneal_steps
        ):
            # DomainBed resets Adam when the penalty coefficient changes sharply.
            optimizer = build_optimizer(model, args)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        objective, last_details = compute_objective(
            algorithm, logits, labels, groups, args, state
        )
        objective.backward()
        optimizer.step()
        state["update_count"] += 1
        running_loss += float(objective.detach()) * labels.numel()
        seen += labels.numel()
    return running_loss / max(seen, 1), last_details, optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    state: dict,
    phase: str,
    epoch: int,
) -> None:
    algorithm_state = {
        key: value.detach().cpu() if isinstance(value, Tensor) else value
        for key, value in state.items()
    }
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "phase": phase,
            "epoch": epoch,
            "algorithm_state": algorithm_state,
        },
        path,
    )


def train(algorithm: str, args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    splits = load_splits(args)
    node_dim, edge_dim = infer_feature_dims(splits["train"])
    model = DrugOODGIN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = build_optimizer(model, args)
    group_algorithms = {"irm", "vrex", "groupdro"}
    loaders = {
        name: make_loader(
            dataset,
            args,
            shuffle=name == "train",
            group_balanced=name == "train" and algorithm in group_algorithms,
        )
        for name, dataset in splits.items()
    }
    erm_pretrain_loader = make_loader(
        splits["train"], args, shuffle=True, group_balanced=False
    )
    state = {"update_count": 0}
    if algorithm == "groupdro":
        state["group_values"] = all_group_values(splits["train"]).to(device)
        state["group_weights"] = torch.ones_like(state["group_values"], dtype=torch.float)
        state["group_weights"] /= state["group_weights"].numel()

    stem = dataset_stem(args)
    run_name = f"{algorithm}_{stem}_seed{args.seed}_{int(time.time())}"
    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_accuracy = -math.inf
    best_epoch = 0
    best_phase = "main"
    stale_epochs = 0
    history = []

    print(f"algorithm={algorithm} dataset={stem} device={device} output={output_dir}")

    pretrain_epochs = args.erm_pretrain_epochs if algorithm != "erm" else 0
    if pretrain_epochs > 0:
        pretrain_state = {"update_count": 0}
        for epoch in range(1, pretrain_epochs + 1):
            train_objective, last_details, optimizer = train_one_epoch(
                model,
                erm_pretrain_loader,
                device,
                optimizer,
                "erm",
                args,
                pretrain_state,
            )
            val_metrics = evaluate(model, loaders["ood_val"], device)
            record = {
                "phase": "erm_pretrain",
                "epoch": epoch,
                "train_objective": train_objective,
                "ood_val": val_metrics,
                **last_details,
            }
            history.append(record)
            if epoch % args.log_every == 0:
                print(
                    f"phase=erm_pretrain epoch={epoch:03d} train={train_objective:.4f} "
                    f"ood_val_acc={val_metrics['accuracy']:.4f} "
                    f"ood_val_auc={val_metrics['roc_auc']:.4f}"
                )
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                best_epoch = epoch
                best_phase = "erm_pretrain"
                save_checkpoint(best_path, model, args, state, best_phase, epoch)

        # The representation is retained; Adam state is reset for the OOD objective.
        optimizer = build_optimizer(model, args)

    for epoch in range(1, args.epochs + 1):
        train_objective, last_details, optimizer = train_one_epoch(
            model, loaders["train"], device, optimizer, algorithm, args, state
        )

        val_metrics = evaluate(model, loaders["ood_val"], device)
        record = {
            "phase": "main",
            "epoch": epoch,
            "train_objective": train_objective,
            "ood_val": val_metrics,
            **last_details,
        }
        history.append(record)
        if epoch % args.log_every == 0:
            print(
                f"phase=main epoch={epoch:03d} train={record['train_objective']:.4f} "
                f"ood_val_acc={val_metrics['accuracy']:.4f} "
                f"ood_val_auc={val_metrics['roc_auc']:.4f}"
            )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_phase = "main"
            stale_epochs = 0
            save_checkpoint(best_path, model, args, state, best_phase, epoch)
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(
                    f"early stopping at main epoch={epoch}; "
                    f"best={best_phase}:{best_epoch}"
                )
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = {name: evaluate(model, loader, device) for name, loader in loaders.items() if name != "train"}
    summary = {
        "algorithm": algorithm,
        "dataset": stem,
        "seed": args.seed,
        "best_phase": best_phase,
        "best_epoch": best_epoch,
        "best_ood_val_accuracy": best_accuracy,
        "metrics": metrics,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
