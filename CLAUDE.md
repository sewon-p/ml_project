# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SUMO 시뮬레이션 기반 **단일 프로브 차량**의 6채널 동역학 시계열(VX, VY, AX, AY, speed, brake)로 교통 밀도(k)와 교통량(q)을 추정하는 ML 프로젝트.
시나리오별 동적 도로 길이(speed_limit×600×1.2)에서 5대 프로브 차량의 300초 시계열을 추출 (20,000 시나리오 × 5 프로브 = 100,000 샘플, 패딩 없음).
XGBoost, LightGBM, CNN1D, LSTM 모델과 Fundamental Diagram 역산 베이스라인을 비교한다.

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Install with SUMO simulation support
pip install -e ".[dev,simulation]"

# Run all tests
pytest

# Run a single test
pytest tests/test_features.py::TestBasicStats -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Generate simulation scenarios
python scripts/generate_scenarios.py --config configs/simulation/scenarios.yaml

# Run SUMO simulation
python scripts/run_simulation.py --config configs/simulation/scenarios.yaml

# Extract features from simulation output
python scripts/extract_features.py --config configs/default.yaml

# Train a model
python scripts/train.py --config configs/default.yaml

# Run a specific experiment
python scripts/run_experiment.py --config configs/experiments/exp1_single_probe_ml.yaml

# Evaluate and generate plots
python scripts/evaluate.py --config configs/default.yaml

# Run full pipeline (simulate → extract → train → evaluate)
python scripts/run_all.py --config configs/default.yaml

# TensorBoard
tensorboard --logdir runs/
```

## Architecture

- `src/features/` — 피처 엔지니어링. `@register_feature` 데코레이터 + `FeatureRegistry`. 입력: trajectory DataFrame (VX, VY, AX, AY, speed, brake)
  - `basic_stats.py` — speed/VX/VY 채널별 mean/std/min/max/percentiles
  - `acceleration.py` — AX/AY 기반 가속도/감속도/jerk 피처
  - `stop_patterns.py` — speed 기반 정지 패턴 (정지 횟수, 비율, 평균 지속시간)
  - `time_series.py` — speed/VX 자기상관, FFT, sample entropy
  - `brake_patterns.py` — brake 횟수, 비율, 평균 지속시간
  - `lateral.py` — VY 분산, 차선변경 추정, 횡방향 에너지
- `src/models/` — 모델 정의. `BaseEstimator` ABC (fit/predict/save/load). XGBoost, LightGBM, CNN1D(6ch), LSTM(6ch), FD baseline
- `src/data/` — 데이터 파이프라인. `TrafficDataset` (tabular), `TimeSeriesDataset` (PyTorch 6채널 시계열)
- `src/training/` — 학습 인프라. Tabular trainer (GroupKFold CV), DL trainer (PyTorch loop), Optuna hyperopt
- `src/simulation/` — SUMO 시뮬레이션. 동적 도로 길이 네트워크, 시나리오 매트릭스, 5-프로브 무작위 선택, 6채널 시계열 추출, Edie ground truth
- `src/evaluation/` — 평가. RMSE/MAE/MAPE/R², 교통상태 분류, SHAP 분석, 결과 집계
- `src/visualization/` — 시각화. FD plot, predicted vs actual, SHAP, 모델 비교
- `src/utils/` — 공통 유틸. config 로딩 (`_base_` 상속), 로깅, seed 관리, 체크포인트 save/load
- `configs/` — YAML 설정. `default.yaml` + 실험별/모델별/피처별 오버라이드
- `scripts/` — CLI 진입점. generate_scenarios, run_simulation, extract_features, train, evaluate, run_all

## Conventions

- **Config-Driven**: 하이퍼파라미터는 YAML 파일로 관리, `_base_`로 상속, 코드에 하드코딩하지 않음
- **Feature Registry**: `@register_feature` 데코레이터로 등록, config의 이름 리스트로 서브셋 선택. 입력 시그니처: `(trajectory: pd.DataFrame, **kwargs) -> float`
- **6-Channel Trajectory**: 모든 피처 함수는 VX, VY, AX, AY, speed, brake 컬럼을 가진 DataFrame을 입력으로 받음
- **5-Probe Extraction**: 시나리오당 200~300초 프로브 후보 선정 → 무작위 5대 → 300~600초 궤적 추출, 패딩 없이 정확히 (6, 300)
- **BaseEstimator ABC**: 새 모델은 `src/models/`에 `BaseEstimator` 서브클래스로 추가
- **GroupKFold by scenario_id**: 학습/검증 split 시 scenario_id 기준으로 그룹 분리 (데이터 누출 방지)
- **SUMO 의존성 격리**: `src/simulation/`에만 traci/sumolib 의존, 나머지 모듈은 독립 실행 가능
- 데이터 I/O는 Parquet 포맷 사용 (`src/data/io.py`)
