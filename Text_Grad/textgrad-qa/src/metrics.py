"""
평가 지표: EM (Exact Match), F1 Score
"""
import string
import unicodedata
from collections import Counter
from typing import List, Union


def normalize_answer(s: str) -> str:
    """답변 정규화: 소문자 변환, 구두점 제거, 공백 정리"""
    s = unicodedata.normalize('NFC', s)
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    return white_space_fix(remove_punc(s.lower()))


def calculate_em(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    Exact Match 계산
    
    Args:
        prediction: 모델의 예측 답변
        ground_truth: 정답 (문자열 또는 문자열 리스트)
    
    Returns:
        1.0 (일치) 또는 0.0 (불일치)
    """
    pred_norm = normalize_answer(prediction)
    
    # ground_truth가 리스트인 경우 처리
    if isinstance(ground_truth, list):
        gt_norm_list = [normalize_answer(gt) for gt in ground_truth]
    else:
        gt_norm_list = [normalize_answer(ground_truth)]
    
    return 1.0 if pred_norm in gt_norm_list else 0.0


def calculate_f1(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    F1 Score 계산 (토큰 단위 중첩도)
    
    Args:
        prediction: 모델의 예측 답변
        ground_truth: 정답 (문자열 또는 문자열 리스트)
    
    Returns:
        F1 점수 (0.0 ~ 1.0)
    """
    # ground_truth가 리스트인 경우, 최대 F1 반환
    if isinstance(ground_truth, list):
        if not ground_truth:
            return 0.0
        return max(calculate_f1(prediction, gt) for gt in ground_truth)
    
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # 공통 토큰 계산
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0 or len(prediction_tokens) == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
