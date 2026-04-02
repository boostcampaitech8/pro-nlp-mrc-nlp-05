"""
평가 지표: EM (Exact Match), F1 Score
"""
import string
import unicodedata
from collections import Counter
from typing import List, Union


def normalize_answer(s: str) -> str:
    if s is None:
        return ""

    s = unicodedata.normalize('NFC', s)
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    return white_space_fix(remove_punc(s.lower()))


def calculate_em(prediction: Union[str, None],
                 ground_truth: Union[str, List[str], None]) -> float:
    """
    Exact Match 계산
    """
    # prediction 방어
    if not prediction:
        return 0.0

    # ground_truth 방어
    if not ground_truth:
        return 0.0

    pred_norm = normalize_answer(prediction)

    if isinstance(ground_truth, list):
        gt_norm_list = [
            normalize_answer(gt) for gt in ground_truth if gt
        ]
    else:
        gt_norm_list = [normalize_answer(ground_truth)]

    if not gt_norm_list:
        return 0.0

    return 1.0 if pred_norm in gt_norm_list else 0.0


def calculate_f1(prediction: Union[str, None],
                 ground_truth: Union[str, List[str], None]) -> float:
    """
    F1 Score 계산 (토큰 단위 중첩도)
    """
    # prediction 방어
    if not prediction:
        return 0.0

    # ground_truth 방어
    if not ground_truth:
        return 0.0

    if isinstance(ground_truth, list):
        valid_gt = [gt for gt in ground_truth if gt]
        if not valid_gt:
            return 0.0
        return max(calculate_f1(prediction, gt) for gt in valid_gt)

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    return (2 * precision * recall) / (precision + recall)