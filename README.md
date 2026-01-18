# 📝 Open-Domain Question Answering


##  대회 소개
일반 기계독해와 달리 질문에 대한 지문이 미리 주어지지 않기 때문에 Knowledge Resource(Wikipedia 문서 더미)에서  
관련 문서를 직접 검색하는 Retriever, 문서를 읽고 답변을 생성하는 Reader가 필요한 태스크

- **Task**: 질문에 적합한 문서를 찾는 **Retriever**와 해당 문서에서 정답을 추출하는 **Reader** 모델 구축
- **Duration**: 2025.12.01 ~ 2025.12.11
- **Evaluation Metric**: Exact Match (EM)

## 🏆 리더 보드
### 🥈 Private Leader Board (2위)
<img width="745" height="113" alt="Image" src="https://github.com/user-attachments/assets/ffab5709-0756-42cf-9945-1e97b064fa85" />

## 👥 팀원 소개

<table align='center'>
  <tr>
    </td>
        <td align="center">
      <img src="https://github.com/Jiho001.png" alt="김지호" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Jiho001">
        <img src="https://img.shields.io/badge/김지호-grey?style=for-the-badge&logo=github" alt="badge 김지호"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/aaronnoah1112.png" alt="노찬민" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/aaronnoah1112">
        <img src="https://img.shields.io/badge/노찬민-grey?style=for-the-badge&logo=github" alt="badge 노찬민"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/HWsong00.png" alt="송현우" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/HWsong00">
        <img src="https://img.shields.io/badge/송현우-grey?style=for-the-badge&logo=github" alt="badge 송현우"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/YangHyunu.png" alt="양현우" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/YangHyunu">
        <img src="https://img.shields.io/badge/양현우-grey?style=for-the-badge&logo=github" alt="badge 양현우"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/hyos0415.png" alt="장효성" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/hyos0415">
        <img src="https://img.shields.io/badge/장효성-grey?style=for-the-badge&logo=github" alt="badge 장효성"/>
      </a>
    </td>
    <td align="center">
      <img src="https://github.com/Sunghae-Cho.png" alt="조성해" width="100" height="100" style="border-radius: 50%;"/><br>
      <a href="https://github.com/Sunghae-Cho">
        <img src="https://img.shields.io/badge/조성해-grey?style=for-the-badge&logo=github" alt="badge 조성해"/>
      </a>
    </td>
  </tr>
</table>

#### 역할 분담

| 이름   | 역할                                                                                                         |
| ------ | ------------------------------------------------------------------------------------------------------------ |
| 김지호 | ...                   |
| 노찬민 | ...     |
| 송현우 | ...        |
| 양현우 | ... |
| 장효성 | ... |
| 조성해 | ...     |

## 개요

> **목표**: 한국어 MRC(Machine Reading Comprehension) 대회 환경에서 형태소 분석기(Kiwi), 개체명 인식(GLiNER), 지식 그래프(유의어 확장)를 결합한 **Hybrid Retrieval**과 추출 모델 및 생성 모델을 조합한 **Compound AI Reader** 시스템을 구축하여 정답 추출 성능(EM)을 극대화함.

---

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [핵심 특징](#핵심-특징)
- [전체 RAG 검색 전략](#전체-rag-검색-전략)
- [플로우차트](#플로우차트)
- [폴더 구조](#폴더-구조)
- [결과 분석](#결과-분석)

---

## 프로젝트 개요
한국어가 조사와 용언 활용이 다양한 교착어라는 언어적 특성을 반영하여, 형태소 기반 토큰화 및 개체명 인식을 활용한 정교한 문서 검색 전략을 설계했습니다. 대규모 사전학습 모델(LLM)의 문맥 이해력과 추출형 모델(Extractive QA)의 형식 안정성을 시스템적으로 결합한 **Compound AI System**을 통해 하드웨어 제약 내에서 최적의 성능을 도출하는 데 목적이 있습니다.

---

## 핵심 특징

### 1) Hybrid Retrieval (Sparse + Dense)
* **Sparse Retrieval**: Kiwi 형태소 분석기로 어간을 정규화하여 용언 활용에 대응하고, GLiNER 모델을 통해 고유명사 등 핵심 정보의 의미 단위를 보존합니다. 또한 '우리말샘' 기반의 지식 그래프를 구축하여 유의어 관계 유형에 따른 가중치 기반 BM25 확장을 수행합니다.
* **Dense Retrieval**: BGE-m3 및 Qwen 임베딩 모델을 활용하여 의미적 유사도를 탐색하며, 데이터 기반 실험을 통해 최적의 리콜(Recall)을 위한 청크 사이즈(256)와 오버랩(128)을 선정했습니다.

### 2) Compound AI Reader System
* **klue/roberta-large**: 한국어 Span 추출 성능이 검급된 모델을 기준으로 삼아 답변의 형식 안정성을 확보합니다.
* **Qwen3**: 추출형 모델의 답변 신뢰도(Confidence)가 낮은 문항에 대해 LLM이 문맥을 재해석하여 보완하는 이원화된 파이프라인을 구축했습니다.
* **Multi-role Prompting**: Solver, Critic, Prompt Engineer 역할을 분리하여 오답 패턴을 분석하고 프롬프트를 자동으로 개선하는 최적화 프로세스를 도입했습니다.

---

## 전체 RAG 검색 전략


1. **FAISS vector storage 속도 최적화**:  storage_context 를 커스텀 구축하여 FAISS의 검색 속도를 최적화함.
2. **reranker**: 코사인 유사도 기반 점수를 함께 출력하여 정렬
3. **토큰화 및 엔티티 분석**: 질문에서 Kiwi로 형태소를 분리하고 GLiNER로 핵심 개체명을 추출합니다.
4. **유의어 가중치 확장**: 지식 그래프를 통해 추출된 유의어에 관계 유형별 가중치(0.2~0.95)를 부여하여 쿼리를 풍부화합니다.
5. **Hybrid Fusion**: Sparse 검색 결과와 Dense 검색 결과를 합집합으로 구성한 뒤, `BGE-reranker-v2-m3`로 재정렬하여 최종 컨텍스트를 선정합니다.

---

## 플로우차트



* **Retrieval** (Sparse + Dense) → **Reranking** → **1차 Reader** (RoBERTa) → **Confidence Check** → (필요 시) **2차 Reader** (LLM) → **최종 답변 생성**
<p align="center">
  <img width="459" height="676" alt="image" src="https://github.com/user-attachments/assets/2c102c61-63d7-4e23-9b89-db216431c442" />
</p>

---

## 폴더 구조
```text
nlp-mrc-project/
│
├── llm/                             # LLM 기반 추론 및 프롬프트 관리
│   ├── configs/                     # LLM 실행을 위한 설정 파일
│   │   ├── config.yaml
│   │   └── prompts/                 # 프롬프트 템플릿 모음
│   │       ├── initial_rules.yaml
│   │       └── system_prompt_template.txt
│   ├── data/                        
│   ├── experiments/               
│   ├── textgrad/                    
│   │   ├── __init__.py
│   │   ├── metrics.py               
│   │   ├── optimizer.py            
│   │   ├── qa_engine.py             
│   │   ├── train.py                 
│   │   └── utils.py                 
│   │
│   ├── llm_inference.py             
│   ├── system_msg_written_by_human.txt  
│   ├── requirements.txt            
│   └── README.md                    
│
├── rag/                             # Retrieval-Augmented Generation 모듈
│   ├── retriever/                   # 다양한 리트리버 구현체
│   │   ├── kiwi_NER_synonym_hybrid.py   # Kiwi + NER + 유의어 확장 하이브리드 리트리버
│   │   ├── kiwi_dense_hybrid.py     # Kiwi + Dense 하이브리드
│   │   ├── NER_dense_hybrid.py      # NER + Dense 하이브리드
│   │   ├── dense_rag_bge.py         # BGE 임베딩 기반 Dense RAG
│   │   ├── dense_rag_qwen.py        # Qwen 임베딩 기반 Dense RAG
│   │   ├── Reranker.py              # Cross-Encoder 리랭커
│   │   └── convert_to_json.py       # 검색 결과 JSON 변환 유틸
│   │
│   └── compress.py                  # 검색 문서 압축/전처리 로직
│
├── roberta/                         # Roberta 기반 Encoder 모델
│   ├── util/
│   │   ├── calculate_recall.py      # Recall 계산 스크립트
│   │   └── ensemble_nbest.ipynb     # n-best 앙상블 실험
│   │
│   ├── arguments.py                 # 학습/추론 인자 정의
│   ├── train.py                     # Roberta 학습 스크립트
│   ├── inference.py                 # Roberta 추론 스크립트
│   ├── retrieval.py                 # Roberta 기반 retrieval
│   ├── precomputed_retrieval.py     # 사전 계산된 retrieval 로딩
│   ├── trainer_qa.py                # QA 전용 Trainer
│   └── utils_qa.py                  # QA 유틸 함수
│
├── ner/                             # NER 관련
│   ├── Kiwi+Ner+유의어확장.ipynb       # Kiwi + NER + 유의어 확장 실험
│   └── Ner+BM25.ipynb               # NER + BM25 실험
├── .gitignore
└── README.md                        # 프로젝트 전체 설명

```


## 결과 분석
## 1. Retrieval 성능 평가 (Recall@k)

지식 그래프 기반 유의어 확장과 Sparse-Dense 결합을 통해 정답 문서 확보 능력을 극대화했습니다. 특히 단독 검색 방식 대비 약 **3~4%p의 Recall 상승**을 기록했습니다. 

| 분석 단위 (k) | Recall 점수 (%) | 비고 |
| :--- | :---: | :--- |
| **Top-5** | **97.50%** | 최종 파이프라인 채택 기준 |
| Top-8 | 98.33% |  |
| Top-10 | 98.75% | |



---

## 2. Reader 모델 성능 평가

### 2.1 LLM 프롬프트 최적화 결과
Multi-role 피드백 루프를 통해 LLM의 정답 추출 성능(EM)을 유의미하게 향상시켰습니다. 

| 모델 명칭 | 정답 추출 방식 | EM Score | F1 Score |
| :--- | :--- | :---: | :---: |
| Qwen3-Instruct | Base Prompt | 0.6542 | -  |
| **Qwen3-Instruct** | **Tuned (Proposed)** | **0.7000** | **0.7860**  |
| Gemma3-27b-it | Base Prompt | 0.6750 | 0.7632  |
| **Gemma3-27b-it** | **Tuned (Proposed)** | **0.7042** | **0.7767**  |

### 2.2 RoBERTa 커리큘럼 러닝 성과
전이 학습과 난이도별 데이터 학습(Curriculum Learning)을 통해 성능을 개선했습니다. 

* **Baseline (전이 학습 미적용)**: 67.50% (EM) / 80.16% (F1) 
* **KorQuad 전이 학습 적용**: 71.67% (EM) / 81.66% (F1) 
***전이 후 커리큘럼 러닝 적용**: **73.75% (EM)** / **83.56% (F1)** 

---

## 3. 최종 대회 성적 (Final Result)

추출형 모델의 안정성과 생성형 모델의 추론 능력을 결합한 **Compound AI System** 앙상블 전략으로 최종 순위를 확보했습니다. 

| 평가 구분 | EM Score | 최종 순위 |
| :--- | :---: | :---: |
| **Public 리더보드** | **77.50%** | **1위**  |
| **Private 리더보드** | **71.11%** | **2위**  |
