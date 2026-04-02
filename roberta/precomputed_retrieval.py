import json
import os
from typing import List, Optional, Union, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value


class PrecomputedRetrieval:
    """
    사전 계산된 RAG top-k 결과를 사용하는 Retrieval 클래스
    """

    def __init__(
        self,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
        precomputed_path: Optional[str] = None,
    ):
        """
        Arguments:
            data_path: 데이터 경로
            context_path: Wikipedia 문서 JSON 파일명
            precomputed_path: 사전 계산된 RAG 결과 JSON 파일 경로
        """
        self.data_path = data_path
        
        # Wikipedia 문서 로드
        print(f"Loading Wikipedia documents from {os.path.join(data_path, context_path)}")
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
        
        print("====wiki 로드 성공 ----!!")
        
        # document_id -> text 매핑
        self.id_to_text = {v["document_id"]: v["text"] for v in wiki_data.values()}
        print(f"Loaded {len(self.id_to_text)} documents")
        
        # 사전 계산된 결과 로드 (옵션)
        self.precomputed_data = None
        if precomputed_path:
            self.load_precomputed(precomputed_path)

    def load_precomputed(self, precomputed_path: str):
        """사전 계산된 RAG 결과 로드"""
        print(f"Loading precomputed retrieval results from {precomputed_path}")
        with open(precomputed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # question_id -> document_ids 매핑으로 변환
        question_ids = data["question_id"]
        document_ids = data["document_id"]
        
        self.precomputed_data = {
            qid: doc_ids
            for qid, doc_ids in zip(question_ids, document_ids)
        }
        print(f"Loaded precomputed results for {len(self.precomputed_data)} questions")

    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset],
        topk: Optional[int] = 5,
        precomputed_path: Optional[str] = None,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        사전 계산된 결과를 사용하여 문서 검색
        
        Arguments:
            query_or_dataset: Dataset (question_id와 question 포함)
            topk: 사용할 문서 개수 (precomputed 결과에서 상위 topk개만 사용)
            precomputed_path: 사전 계산된 결과 파일 (기존과 다른 파일 사용 시)
            
        Returns:
            pd.DataFrame with columns: id, question, context, [answers, original_context]
        """
        # precomputed_path가 주어지면 새로 로드
        if precomputed_path and precomputed_path != getattr(self, '_last_precomputed_path', None):
            self.load_precomputed(precomputed_path)
            self._last_precomputed_path = precomputed_path
        
    
        if self.precomputed_data is None:
            raise ValueError(
                "No precomputed data loaded. "
                "Please provide precomputed_path in __init__ or retrieve()"
            )
        
        if isinstance(query_or_dataset, Dataset):
            return self._retrieve_bulk(query_or_dataset, topk)
        else:
            raise NotImplementedError(
                "Single query retrieval not supported. "
                "Use Dataset with question_id"
            )

    def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        """Dataset에 대한 검색"""
        total = []
        
        missing_count = 0
        for example in dataset:
            question_id = example["id"]
            question = example["question"]
            
            # 사전 계산된 document_ids 가져오기
            if question_id not in self.precomputed_data:
                print(f"Warning: Question ID {question_id} not found in precomputed data")
                missing_count += 1
                # 빈 context 사용
                doc_ids = []
            else:
                doc_ids = self.precomputed_data[question_id][:topk]
            
            # Document IDs를 실제 text로 변환
            contexts = []
            for doc_id in doc_ids:
                if doc_id in self.id_to_text:
                    contexts.append(self.id_to_text[doc_id])
                else:
                    print(f"Warning: Document ID {doc_id} not found in wiki data")
            
            # 여러 문서를 공백으로 연결
            context = " ".join(contexts) if contexts else ""
            
            tmp = {
                "question": question,
                "id": question_id,
                "context": context,
            }
            
            # Ground truth 있으면 추가
            if "context" in example.keys() and "answers" in example.keys():
                tmp["answers"] = example["answers"]
            
            total.append(tmp)
        
        if missing_count > 0:
            print(f"Warning: {missing_count} questions not found in precomputed data")
        
        return pd.DataFrame(total)

    def retrieve_with_scores(
        self,
        dataset: Dataset,
        topk: Optional[int] = 5,
        precomputed_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        검색 결과와 함께 document IDs도 반환
        (디버깅이나 분석 용도)
        """
        if precomputed_path:
            self.load_precomputed(precomputed_path)
        
        if self.precomputed_data is None:
            raise ValueError("No precomputed data loaded")
        
        total = []
        
        for example in dataset:
            question_id = example["id"]
            question = example["question"]
            
            doc_ids = self.precomputed_data.get(question_id, [])[:topk]
            
            contexts = [
                self.id_to_text.get(doc_id, "")
                for doc_id in doc_ids
            ]
            
            context = " ".join([c for c in contexts if c])
            
            tmp = {
                "question": question,
                "id": question_id,
                "context": context,
                "retrieved_doc_ids": doc_ids,  # 추가 정보
            }
            
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            
            total.append(tmp)
        
        return pd.DataFrame(total)


def run_precomputed_retrieval(
    datasets: DatasetDict,
    training_args,
    data_args,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
    precomputed_path: str = None,
) -> DatasetDict:
    """
    사전 계산된 RAG 결과를 사용하여 retrieval 수행
    
    Arguments:
        datasets: train/validation/test dataset
        training_args: TrainingArguments
        data_args: DataTrainingArguments
        data_path: 데이터 경로
        context_path: Wikipedia 문서 파일명
        precomputed_path: 사전 계산된 RAG 결과 JSON 경로
        
    Returns:
        DatasetDict with retrieved contexts
    """
    from datasets import Dataset
    
    # PrecomputedRetrieval 초기화
    retriever = PrecomputedRetrieval(
        data_path=data_path,
        context_path=context_path,
        precomputed_path=precomputed_path,
    )
    
    # Retrieval 수행
    topk = getattr(data_args, 'top_k_retrieval', 5)
    df = retriever.retrieve(
        datasets["validation"],
        topk=topk,
    )
    
    # Features 정의
    if training_args.do_predict:
        f = Features({
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
    elif training_args.do_eval:
        f = Features({
            "answers": Sequence(
                feature={
                    "text": Value(dtype="string", id=None),
                    "answer_start": Value(dtype="int32", id=None),
                },
                length=-1,
                id=None,
            ),
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
    
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


if __name__ == "__main__":
    # 테스트 코드
    from datasets import load_from_disk
    
    # 데이터 로드
    data_path = "../data"
    dataset = load_from_disk(f"{data_path}/train_dataset")
    
    # PrecomputedRetrieval 사용
    retriever = PrecomputedRetrieval(
        data_path=data_path,
        context_path="wikipedia_documents.json",
        precomputed_path="./document/rag_top5.json",
    )
    
    # 검색 수행
    df = retriever.retrieve(dataset["validation"][:10], topk=5)
    
    print(df.head())
    print(f"\nRetrieved {len(df)} examples")
    print(f"Context length example: {len(df.iloc[0]['context'])} characters")