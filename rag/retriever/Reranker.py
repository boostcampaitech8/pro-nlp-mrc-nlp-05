from sentence_transformers import CrossEncoder 
import torch
from typing import List, Dict


RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    def rerank(self, query: str, docs: List[Dict], doc_id, top_k: int = 5) -> List[Dict]:
        """
        query와 docs[{'text': ..., ...}]를 받아, score 기준으로 다시 정렬해서 top_k만 반환합니다.
        """
        if not docs:
            return []

        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)  # shape (len(docs),)
        scored_docs = list(zip(docs, scores))
        scored_id = list(zip(doc_id, scores)) 

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        scored_id.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k], scored_id[:top_k]