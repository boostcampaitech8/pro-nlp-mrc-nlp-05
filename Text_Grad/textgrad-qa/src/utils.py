"""
데이터 로더 및 유틸리티
"""
import json
from typing import List, Dict, Any
from pathlib import Path


def load_json(file_path: str) -> Any:
    """JSON 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    """JSON 파일 저장"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_wikipedia_documents(file_path: str) -> Dict[str, str]:
    """
    Wikipedia 문서 로드
    
    Returns:
        {doc_id: content} 형태의 딕셔너리
    """
    data = load_json(file_path)
    return {item['id']: item['content'] for item in data}


def format_qa_input(question: str, context: str) -> str:
    """
    QA 입력 포맷팅
    
    Args:
        question: 질문 문자열
        context: 문맥 문자열
    
    Returns:
        포맷된 입력 문자열
    """
    return f"""[Context]
{context}

[Question]
{question}

위 Context에서 Question에 대한 정답을 추출하십시오."""


def save_prompt(prompt: str, save_path: str):
    """프롬프트 텍스트 저장"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(prompt)


def load_prompt(file_path: str) -> str:
    """프롬프트 텍스트 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
