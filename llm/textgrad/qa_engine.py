"""
QA Engine: Solver 역할
주어진 시스템 프롬프트로 QA 수행
"""
import re
import json
from typing import Optional


class QA_Engine:
    """질문-답변 엔진"""
    
    def __init__(self, llm_engine, system_prompt: str):
        """
        Args:
            llm_engine: llama-cpp-python의 Llama 인스턴스
            system_prompt: 시스템 프롬프트 (규칙 포함)
        """
        self.llm = llm_engine
        self.system_prompt = system_prompt
    
    def predict(self, formatted_prompt: str, verbose: bool = False) -> str:
        """
        질문에 대한 답변 생성
        
        Args:
            formatted_prompt: 포맷된 사용자 입력 (질문 + 문맥)
            verbose: 응답 전체 출력 여부
        
        Returns:
            추출된 답변 문자열
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': formatted_prompt}
        ]
        
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=4096,
                temperature=0.0
            )
            response_text = output['choices'][0]['message']['content']
            
            if verbose:
                print(f"[QA Response] {response_text}")
            
            # JSON 파싱 시도
            try:
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return str(data.get("extracted_answer", ""))
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 전체 텍스트 반환
                return response_text.strip()
        
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return ""
    
    def update_prompt(self, new_prompt: str):
        """시스템 프롬프트 업데이트"""
        self.system_prompt = new_prompt
