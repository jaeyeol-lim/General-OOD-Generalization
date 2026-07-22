# General baselines for DrugOOD IC50

이 디렉터리는 동일한 4-layer GIN 백본으로 ERM, IRM, V-REx, GroupDRO를 실행한다. 호스트와 컨테이너에서 다음 전처리 캐시 경로를 자동 탐색한다.

```text
/home/jylim/project/Graph-OOD-Lab/data/DrugOOD/
/workspace/Graph-OOD-Lab/data/DrugOOD/
  drugood_lbap_core_ic50_{assay,scaffold,size}_{train,ood_val,ood_test}.pt
```

DrugOOD IC50은 이 문서에 정리된 EC50 실험 규약을 따르되, 문서에 없는 공식 설정은 IC50 config를 우선한다. OOD validation accuracy로 best checkpoint를 선택하며 accuracy와 ROC-AUC를 함께 기록한다.

## 단일 실행

```bash
cd /home/jylim/project/baselines/general

python3 erm.py --domain assay
python3 irm.py --domain assay --penalty-weight 1
python3 vrex.py --domain assay --penalty-weight 1
python3 groupdro.py --domain assay --step-size 0.1
```

기본 출력은 `outputs/` 아래에 저장되며, 각 실행은 `best.pt`, `history.json`, `summary.json`을 생성한다. `--output-dir`로 위치를 지정할 수 있다.

## 하이퍼파라미터 탐색

```bash
# 명령만 확인
python3 sweep_erm.py --domains assay scaffold size --seeds 1 2 3 4 --dry-run
python3 sweep_irm.py --domains assay scaffold size --seeds 1 2 3 4 --dry-run

# 실제 실행
python3 sweep_erm.py --domains assay scaffold size --seeds 1 2 3 4
python3 sweep_irm.py --domains assay scaffold size --seeds 1 2 3 4
python3 sweep_vrex.py --domains assay scaffold size --seeds 1 2 3 4
python3 sweep_groupdro.py --domains assay scaffold size --seeds 1 2 3 4
```

탐색 범위는 다음과 같다.

| 방법 | 인자 | 값 |
|---|---|---|
| IRM | `--penalty-weight` | `{1e-2, 1e-1, 1, 1e1}` |
| V-REx | `--penalty-weight` | `{1e-2, 1e-1, 1, 1e1}` |
| GroupDRO | `--step-size` | `{1.0, 1e-1, 1e-2}` |

탐색 스크립트가 모르는 추가 학습 인자는 그대로 각 학습 파일로 전달된다.

```bash
python3 sweep_irm.py --domains assay --seeds 1 -- --epochs 50 --batch-size 64
```

`--max-parallel`의 기본값은 GPU 메모리 충돌을 피하기 위해 `1`이다. 병렬 실행 시에는 각 프로세스가 동일한 `--device`를 사용한다는 점에 유의한다.

## 공식 규약과 목적함수 검증

- ERM은 sample mean cross-entropy를 사용한다.
- IRM과 V-REx는 환경별 mean risk를 동일 가중 평균한다.
- GroupDRO는 전체 그룹의 persistent adversarial weight를 유지한다.
- IRM, V-REx, GroupDRO 학습 배치는 서로 다른 4개 그룹을 균등하게 포함한다.
- IRM/V-REx는 500 optimizer step 뒤 탐색 대상 penalty weight를 적용하고 Adam을 reset한다.
- 비-ERM 방법은 standard shuffled batch로 ERM 10 epochs pretraining 후 group-balanced main training을 수행한다.
- 공통 기본값은 Adam, learning rate `1e-3`, 최대 50 main epochs, batch size 64, early-stopping patience 10이다.
- sweep 기본 seed는 `1 2 3 4`이며 완료 후 `sweeps/<method>/aggregate.json`에 mean/std를 자동 집계한다.

수식 회귀 테스트:

```bash
python3 -m unittest -v test_objectives.py
```

## General/Refined subset

폴더명 `general`은 일반적인 OOD 베이스라인 묶음을 뜻한다. DrugOOD의 noise subset을 바꾸려면 `--subset general` 또는 `--subset refined`를 사용할 수 있지만, 해당 이름의 `.pt` 캐시를 먼저 생성해야 한다.

## 주요 공통 옵션

```text
--domain {assay,scaffold,size}
--subset {core,general,refined}
--data-root PATH
--device {auto,cpu,cuda,cuda:N}
--epochs N
--batch-size N
--lr FLOAT
--weight-decay FLOAT
--patience N
--erm-pretrain-epochs N
--num-workers N
```

## 의존성

현재 시스템 기본 `python3`에는 학습 의존성이 설치되어 있지 않다. 기존 DrugOOD 실행 환경을 활성화하거나 다음 패키지를 설치한 환경에서 실행한다.

```bash
python3 -m pip install -r requirements.txt
```
