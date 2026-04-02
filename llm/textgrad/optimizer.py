"""
Prompt Optimizer: Critic + Optimizer 역할
1. Critic: 오답 분석 및 피드백 생성
2. Optimizer: 피드백 기반 프롬프트 재작성
"""
import random
import re
from typing import List, Dict
from omegaconf import DictConfig

class PromptOptimizer:
    """프롬프트 최적화 엔진"""
    
    def __init__(
        self,
        qa_engine,
        llm_engine,
        sample_ratio: float = 0.2,
        max_rules: int = 7,
        evaluation_principles: List[str] = None
    ):
        """
        Args:
            qa_engine: QA_Engine 인스턴스
            llm_engine: LLM 엔진 (피드백/최적화용)
            sample_ratio: 오답 샘플링 비율
            max_rules: 최대 규칙 개수
            evaluation_principles: 평가 기준 (6가지 원칙)
        """
        self.model = qa_engine
        self.llm = llm_engine
        self.sample_ratio = sample_ratio
        self.max_rules = max_rules
        self.principles = evaluation_principles or self._default_principles()
    
    def _default_principles(self) -> List[str]:
        """기본 평가 원칙"""
        return [
            "원칙 1. 질문이 묻는 대상만 추출",
            "원칙 2. 동어반복 금지",
            "원칙 3. 특수문자/기호 제거",
            "원칙 4. 숫자 정확성",
            "원칙 5. 고유명사 정확성",
            "원칙 6. 조사 제거"
        ]
    
    def _sample_errors(self, errors: List[Dict]) -> List[Dict]:
        """오답 샘플링 (일반화를 위해)"""
        sample_size = int(len(errors) * self.sample_ratio)
        sample_size = max(1, min(sample_size, len(errors)))
        return random.sample(errors, sample_size)
    
    def generate_feedback_batch(self, errors: List[Dict]) -> str:
        """
        Critic: 배치 피드백 생성
        
        Args:
            errors: 오답 케이스 리스트
                [{
                    'question': str,
                    'context': str,
                    'pred': str,
                    'gt': str (or list)
                }, ...]
        
        Returns:
            일반화된 피드백 텍스트
        """
        if not errors:
            return ""
        
        sampled = self._sample_errors(errors)
        print(f"   → Sampled {len(sampled)}/{len(errors)} errors for feedback")
        
        # 케이스별 상세 정보 구성
        cases_text = ""
        for i, e in enumerate(sampled):
            gt_text = e['gt'] if isinstance(e['gt'], str) else e['gt'][0]
            
            # 할루시네이션 체크
            norm_ctx = e['context'].replace(" ", "").replace("\n", "")
            norm_pred = e['pred'].replace(" ", "").replace("\n", "")
            is_hallucination = norm_pred not in norm_ctx
            
            cases_text += f"""
[Case {i+1}]
- 질문: {e['question']}
- 정답: {gt_text}
- 예측: {e['pred']}
- 환각 여부: {"예" if is_hallucination else "아니오"}
"""
        
        # Critic 프롬프트
        critique_prompt = f"""당신은 QA 시스템의 오류를 분석하는 언어학 전문가입니다.
제공된 '모델이 틀린 케이스'를 분석하여, 아래 [평가 기준]에 의거해 오류 원인을 진단하고 수정 지침을 내리십시오.

[평가 기준]

{self.principles}

[평가 기준 끝]

다음은 모델이 틀린 케이스들입니다:
###
{cases_text}
###

**수행 과제:**
위 케이스들을 분석하여 아래 형식으로 출력하십시오.

**출력 형식 (반드시 준수):**
각 케이스에 대해 아래 포맷을 엄격히 따르십시오.

[Case N]
- 차이: [모델의 오답] vs [정답] 간의 텍스트 차이 분석
- 원인: **[위반 원칙 번호]**를 먼저 쓰고, 해당 원칙에 위배되는 이유를 설명
- 지시: 정답을 도출하기 위해 모델이 수정해야 할 구체적인 행동

---

**마지막 공통 패턴 요약:**
위 케이스 분석을 바탕으로, 가장 빈번하게 무시된 원칙과 해결책 3가지를 도출하십시오.
1. [위반 원칙/문제점] -> [해결을 위한 지시사항]
2. [위반 원칙/문제점] -> [해결을 위한 지시사항]
3. [위반 원칙/문제점] -> [해결을 위한 지시사항]
"""
        
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": critique_prompt}],
            temperature=0.7
        )
        
        return response['choices'][0]['message']['content'].strip()
    
    def step(self, error_batch: List[Dict]):
        """
        Optimizer: 피드백 기반 프롬프트 업데이트
        
        Args:
            error_batch: 오답 케이스 배치
        """
        if not error_batch:
            return
        
        print(f"\n[Optimizer] Processing {len(error_batch)} errors...")
        
        # 1단계: Critic - 배치 피드백 생성
        summarized_feedback = self.generate_feedback_batch(error_batch)
        print(f"   → Feedback generated\n")
        
        # 2단계: Optimizer - 프롬프트 재작성
        current_prompt = self.model.system_prompt
        
        optimization_prompt = f"""당신은 NLP 정보 추출(Information Extraction) 최적화 전문 프롬프트 엔지니어입니다.
당신의 임무는 주어진 [피드백]을 분석하여, EM(Exact Match) 점수를 100점으로 만들기 위해 [현재 시스템 프롬프트]를 재작성하는 것입니다.

[현재 시스템 프롬프트]
{current_prompt}

[검증 데이터에서의 피드백 (Critical Feedback)]
{summarized_feedback}

[프롬프트 재작성 가이드라인]

1. 규칙의 추상화 및 일반화 (Rule Abstraction):
   - 피드백의 개별 사례에 집착하지 말고, **오류의 원인(Error Type)**을 분석하여 일반화된 규칙을 도출하십시오.
   - 유사한 오류들은 하나의 강력한 상위 규칙으로 통합하십시오.
   - 정답 추출 원칙은 **{self.max_rules}개 이하**로 제한
   - 특정 단어 나열 금지, 모든 도메인에 적용 가능하게 작성

2. Few-Shot 예시 (Crucial):
   - 피드백에서 지적된 오류 케이스를 해결할 수 있는 일반화된 **새로운 Few-Shot 예시(Input-Output 쌍)** 최대 2개를 포함하십시오.

3. 출력 포맷 엄격화:
   - 모델이 추론 과정(Evidence)과 결과(Answer)를 분리하도록 아래 JSON 포맷을 강제하십시오.

출력에는 설명, 요약, 표, 제목, 마크다운을 절대 포함하지 마십시오.
오직 "시스템 프롬프트 본문 텍스트"만 출력하십시오.

[출력 형식]
Evidence: "정답이_포함된_문장"
{{"extracted_answer": "핵심_정답"}}
"""
        
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": optimization_prompt}],
            temperature=0.5
        )
        
        new_prompt = response['choices'][0]['message']['content'].strip()
        
        # 프롬프트 검증
        print(f"🔍 Generated prompt (len={len(new_prompt)})")
        has_format = any(x in new_prompt.lower() for x in ["extracted_answer", "evidence"])
        
        if len(new_prompt) > 100 and has_format:
            self.model.update_prompt(new_prompt)
            print("✅ Prompt updated!")
        else:
            print(f"⚠️ Validation failed. len={len(new_prompt)}, has_format={has_format}")
    
    def _format_principles(self) -> str:
        if isinstance(self.principles, DictConfig):
            # YAML 구조 파싱하여 텍스트 생성
            principles_text = ""
            for idx, (key, principle) in enumerate(self.principles.items(), 1):
                principles_text += f"원칙 {idx}.\n{principle.description}\n\n"
                # 예시들 추가...
            return principles_text
