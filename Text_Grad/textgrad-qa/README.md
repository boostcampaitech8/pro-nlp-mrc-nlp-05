# TextGrad-QA Optimization

**Iterative Prompt Refinement for Korean QA Systems**  
Solver-Critic-Optimizer 아키텍처 기반 프롬프트 자동 최적화

---
**3가지 근본 문제:**

| 문제 | 원인 | 영향 |
|------|------|------|
| **1. EM은 미분 불가능** | 0 or 1 이산값 | 그래디언트 계산 불가 |
| **2. 특정 케이스 과적합** | 개별 오류에 직접 반응 | "냉대기후권" 학습 시 "열대기후권" 미적용 |
| **3. 특수문자 처리 실패** | 규칙 기반 접근 불가 | 《》, '', "" 등 도메인 의존적 패턴 |

---

### 해결책: Rule-Based TextGrad

| 구성 요소 | TextGrad (원본) | 우리 방식 | 개선 효과 |
|-----------|----------------|----------|-----------|
| **손실 함수** | Differentiable Loss (CrossEntropy 등) | **EM + F1 Score** | 이산 지표 직접 활용 |
| **역전파 (Backward)** | `∂L/∂prompt` 계산 | **Critic Agent** (LLM 기반 오류 분석) | 언어적 패턴 추출 |
| **최적화 방향** | Gradient Descent | **Rule Abstraction** (일반화된 규칙 도출) | 도메인 독립적 학습 |
| **배치 처리** | 개별 오류마다 업데이트 | **샘플링 20%** → 배치 피드백 | LLM 호출 90% 감소 |
| **일반화 전략** | 특정 단어/케이스 학습 | **6가지 평가 원칙** 고정 | 과적합 방지 |
| **프롬프트 발산 방지** | 없음 | **규칙 개수 제한 (5~7개)** + 자동 롤백 | Prompt Drift 제어 |

---
---
## 개요

이 프로젝트는 **EM(Exact Match) 기반 QA 태스크**를 위한 프롬프트 자동 최적화 시스템입니다.  
TextGrad의 개념에서 영감을 받아, **규칙 기반 추상화**와 **배치 피드백**을 통해 일반화 성능을 극대화합니다.

### 핵심 특징

- **Solver-Critic-Optimizer 아키텍처**
- **샘플링 기반 일반화** (오답의 20%만 분석)
- **6가지 평가 원칙** (동어반복, 조사 처리, 특수문자 등)
- **Hydra 설정 관리** (실험 재현성)
- **자동 롤백** (성능 저하 시)

---

### 핵심 관찰

실험 중 발견한 흥미로운 패턴:

```
질문: "캐나다의 기후는?"
모델: "캐나다는 냉대기후권에 속하며..." ✅ 정답 문장은 정확히 찾음
     → "냉대기후권" ❌ 하지만 "권" 같은 불필요한 단어 포함
질문: "액운공사에 대한 설명이 나와있는 저서는?"
모델: "대순경전" ❌ , 〈대순경전〉 ✅ 정답 단어를 정확히 찾았으나 <>를 누락함.
```

```
일관적이지 않은 정답 기준:
```

```
- 우선 2025년 말 현실적으로, QA테스크를 Bert와 같은 인코더 모델이 아닌 디코더 모델...
MRC 테스크 특히 평가지표가 EM이라 불리했음에도 현실적인 이유로 디코더 모델을 이용해 도전하였습니다.
- 관찰 결과 디코더 모델이 Instruction(Question)에서 요구한 정답이 포함된 문장(Span)을 매우 정밀하게 추출하는 것을 보았습니다.
따라서 모델이 정확하게 추출한 Span으로 부터 정확히 텍스트를 추출하는 것을 통제할 수 있지 않을까 하여, TextGrad를 시도해보았습니다.
이떄, 원 논문인 TextGrad는 프롬프트를 ... 손실함수, 역전파
그러나 평가지표가 EM이었고, <>,()와 같이 ... 일반화가 불가능한 문제가 있었습니다. 따라서 손실함수를 EM,F1-score로 , Backward를 Critic이라는 에이전트로 ...



## 🏗️ 아키텍처

```
┌──────────────┐
│   Solver     │  ← QA_Engine (현재 프롬프트로 추론)
└──────┬───────┘
       │ 오답 수집
       ↓
┌──────────────┐
│    Critic    │  ← 샘플링(20%) + 6가지 원칙 기반 오류 분석
└──────┬───────┘
       │ 일반화된 피드백
       ↓
┌──────────────┐
│  Optimizer   │  ← 프롬프트 재작성 (5~7개 규칙으로 제한)
└──────┬───────┘
       │ 업데이트
       ↓
    (반복)
```

---

## 📂 프로젝트 구조

```
textgrad-qa/
├── configs/
│   ├── config.yaml                 # 메인 설정
│   └── prompts/
│       ├── system_prompt_template.txt
│       └── initial_rules.yaml
├── src/
│   ├── train.py                    # 메인 실행 스크립트
│   ├── qa_engine.py                # Solver
│   ├── optimizer.py                # Critic + Optimizer
│   ├── metrics.py                  # EM, F1 계산
│   └── utils.py                    # 데이터 로더
├── data/
│   └── wikipedia_documents.json
├── experiments/                    # 실험 결과 저장
├── requirements.txt
└── README.md
```

---

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
cd textgrad-qa

# CUDA 지원 llama-cpp-python 설치 (GPU 사용 시)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# Wikipedia 문서를 data/ 폴더에 배치
# KorQuAD 데이터셋은 자동 다운로드됨
```

### 3. 실행

```bash
cd src
python train.py
```

### 4. 설정 변경 (Hydra)

```bash
# 에포크 수 변경
python train.py optimizer.num_epochs=5

# 샘플링 비율 변경
python train.py critic.sample_ratio=0.3

# 실험 이름 지정
python train.py experiment.name=exp_v2
```

---

## ⚙️ 주요 설정

### `configs/config.yaml`

```yaml
# 모델 설정
model:
  repo_id: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
  n_gpu_layers: -1  # -1 = 모든 레이어 GPU 사용

# Critic 설정
critic:
  sample_ratio: 0.2  # 오답의 20%만 샘플링

# Optimizer 설정
optimizer:
  num_epochs: 10
  max_consecutive_failures: 3  # 3회 연속 실패 시 조기 종료

# 프롬프트 설정
prompt:
  max_rules: 7  # 최대 규칙 개수
```

---

## 📊 실험 결과

실험 후 `experiments/` 폴더에 다음 파일들이 생성됩니다:

```
experiments/
├── prompts/
│   ├── epoch_01_prompt.txt
│   ├── epoch_02_prompt.txt
│   └── ...
├── logs/
│   ├── epoch_01_failures.json
│   └── ...
├── best_prompt.txt               # 최고 성능 프롬프트
└── summary.json                  # 실험 요약
```

---

## 🔧 커스터마이징

### 1. 초기 규칙 변경

`configs/prompts/initial_rules.yaml` 수정:

```yaml
rules:
  - "규칙 1: ..."
  - "규칙 2: ..."
```

### 2. 평가 원칙 변경

`configs/config.yaml`의 `critic.evaluation_principles` 수정

### 3. 템플릿 변경

`configs/prompts/system_prompt_template.txt` 수정

---

## 📈 성능 지표

- **EM (Exact Match)**: 정답과 완전히 일치하는 비율
- **F1 Score**: 정답과의 토큰 중첩도
- **Failure Rate**: 오답 비율

---

## 🤝 기여

이 프로젝트는 다음에서 영감을 받았습니다:
- [TextGrad](https://github.com/zou-group/textgrad) - Automatic Differentiation via Text
- KorQuAD - Korean QA Dataset

---

## 📄 라이선스

MIT License

---

## 🙋‍♂️ 문의

이슈나 질문은 GitHub Issues에 남겨주세요.

---

## 📚 참고 자료

### TextGrad vs 현재 방식

| 항목 | TextGrad | 현재 방식 |
|------|----------|----------|
| 최적화 방법 | Gradient Descent | Rule-Based Refinement |
| 일반화 | 특정 케이스 학습 | 패턴 추상화 |
| EM 적합성 | ⚠️ 보통 | ✅ 높음 |

### Workflow

```python
for epoch in range(NUM_EPOCHS):
    # 1. Solver: 전체 데이터 평가
    predictions = [qa_model.predict(x) for x in data]
    
    # 2. Critic: 오답 샘플링 + 피드백 생성
    errors = [x for x in predictions if em(x) == 0]
    feedback = critic.analyze(sample(errors, 20%))
    
    # 3. Optimizer: 프롬프트 재작성
    new_prompt = optimizer.rewrite(current_prompt, feedback)
    
    # 4. 롤백 (필요 시)
    if performance < best:
        qa_model.prompt = best_prompt
```

---

**Happy Optimizing! 🎯**
