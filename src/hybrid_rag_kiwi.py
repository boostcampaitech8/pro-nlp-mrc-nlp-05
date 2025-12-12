from datasets import load_from_disk
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import numpy as np
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

print("시작!")
HF_TOKEN = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN 
login(token=HF_TOKEN)
import json

# wiki data load
with open('/data/ephemeral/home/data/wikipedia_documents.json') as f:
    wiki_data = json.load(f)
id_to_title = {v["document_id"]: v["title"] for v in wiki_data.values()}


train_set_dir = "/data/ephemeral/home/data/train_dataset/"
dataset = load_from_disk(train_set_dir)


# --- 모델 로드 설정 ---
GEMMA_MODEL_NAME = "google/gemma-3-4b-it"  # 메모리 효율성을 위해 4b 대신 9b를 예시로 사용 (사용자 환경에 따라 변경 가능)
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# --- LLM 및 Tokenizer 로드 ---
def load_gemma():
    """Gemma 모델과 토크나이저를 로드합니다."""
    # Q: Gemma 3-4b-it 사용 예정이었는데, 현재는 Gemma 2-9b-it을 사용하려 합니다.
    # A: VRAM 상황에 따라 모델 이름을 적절히 변경하여 사용하세요.
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_NAME,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model

documents: List[Document] = []
for doc_id, data in wiki_data.items():
    # 'text' 필드를 문서 내용으로 사용
    documents.append(
        Document(
            text=data['text'],
            metadata={
                "document_id": data['document_id'],
                "title": data['title'],
                "corpus_source": data['corpus_source']
            }
        )
    )
    
# 3. 문서 청킹 (Node 생성)
# SentenceSplitter는 문장 단위 분할을 기본으로 하면서, 
# 최종 청크 크기를 chunk_size=512로 제한합니다.
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# nodes에는 작은 텍스트 청크(TextNode)들이 리스트 형태로 담깁니다.
nodes: List[TextNode] = splitter.get_nodes_from_documents(documents)

print(f"원본 문서 개수: {len(documents)}개")
print(f"생성된 청크(Node) 개수: {len(nodes)}개")
print(f"첫 번째 청크 텍스트 예시: {nodes[0].get_content()[:100]}...")

print(torch.cuda.is_available())

embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# faiss vs

dummy_emb = embed_model.get_text_embedding("dim 체크용")
dim = len(dummy_emb)
faiss_index = faiss.IndexFlatIP(dim) 
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("VectorStoreIndex 시작")
vector_index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)
print("끝")
# --- 3. Reranker (사용자 정의) ---
from sentence_transformers import CrossEncoder 

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
    
    
reranker = Reranker()

print("llm setting 시작")
tokenizer, model = load_gemma()
gemma_llm = HuggingFaceLLM(
    # model_name을 지정할 필요가 없거나, 명시적으로 지정해도 model/tokenizer 인자가 우선됩니다.
    model=model,        # 이미 로드된 PyTorch 모델 객체
    tokenizer=tokenizer,  # 이미 로드된 Tokenizer 객체
    #device="cuda" if torch.cuda.is_available() else "cpu",
    # device_map="auto" 등은 이미 model 로드 시 적용되었으므로 LlamaIndex LLM에서는 불필요
    
    # 템플릿 처리 방식 등 LlamaIndex 관련 설정만 추가
    context_window=8192, # 예시: Gemma의 Context Window 설정 (필요에 따라)
)

# 3. LlamaIndex 설정에 적용
Settings.llm = gemma_llm
print("llm setting 끝")

retriever = vector_index.as_retriever(similarity_top_k=50)


print("키위 세팅 시작")
from functools import partial
from kiwipiepy import Kiwi

kiwi = Kiwi()

def tokenize_kiwi(
    text: str,
    kiwi: Kiwi,
    tag_include: List[str],
    text_type: str,
    top_n: int,
    score_threshold: float = 1.05,
) -> list[str]:
    try:
        # 토큰화할 텍스트가 문서일 때
        if text_type == "corpus":
            analyzed = kiwi.analyze(text, top_n=top_n + len(text) // 200)  # [[(분석결과1, 점수), (분석결과2, 점수), ... ]]

            # 분석 결과가 비어있거나 토큰이 없는 경우
            if not analyzed:
                return _fallback_tokenize(text)
            
            num_candi = 1
            # 1위 점수 기준 threshold 이내의 점수이면 num_candi += 1
            while (
                num_candi < len(analyzed)
                and analyzed[num_candi][1] > score_threshold * analyzed[0][1]
            ):
                num_candi += 1
                
        # 토큰화할 텍스트가 쿼리일 때
        elif text_type == "query":
            analyzed = kiwi.analyze(text, top_n=top_n) 

            # 분석 결과가 비어있거나 토큰이 없는 경우
            if not analyzed:
                return _fallback_tokenize(text)
            
            num_candi = 3

        # 모든 후보의 (form, tag) 리스트
        all_tokenized = [
            (t.form, t.tag)
            for nc in range(num_candi)
            for t in analyzed[nc][0]
        ]

        # 중복제거
        unique_tokenized = set(all_tokenized)

        filtered = [
            f"{form}/{tag}"
            for form, tag in unique_tokenized
            if tag in tag_include
        ]
        
        return filtered if filtered else _fallback_tokenize(text)
    
    except Exception:
        return _fallback_tokenize(text)
    

def _fallback_tokenize(text: str) -> list[str]:
    """Kiwi 실패 시 단순 whitespace + 문자 기반 토큰화"""
    import re
    # 공백 분리 + 알파벳/숫자/기타 유니코드 단어 추출
    tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
    return [t for t in tokens]




from typing import List, Optional, Callable
import bm25s
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
from kiwipiepy import Kiwi
import json

class KiwiBM25Retriever(BaseRetriever):
    """Kiwipiepy 토크나이저를 사용하는 한국어 BM25 Retriever (bm25s 기반)"""

    def __init__(
        self,
        nodes: List[BaseNode],
        similarity_top_k: int,
        corpus_tokenizer: Optional[Callable[[str], List[str]]],
        query_tokenizer: Optional[Callable[[str], List[str]]]
    ) -> None:
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k

        # tokenizer가 없으면 기본 Kiwi tokenizer 사용
        if corpus_tokenizer is None:
            kiwi = Kiwi()
            self._tokenizer = lambda text: [f'{t.form}/{t.tag}' for t in kiwi.tokenize(text)]
        if query_tokenizer is None:
            query_tokenizer = corpus_tokenizer
        
        self._corpus_tokenizer = corpus_tokenizer
        self._query_tokenizer = query_tokenizer
        
        # 코퍼스 토크나이징 → index
        corpus_tokens = [self._corpus_tokenizer(node.text) for node in nodes]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = [self._query_tokenizer(query)]

        '''
        results, scores shape = (쿼리 개수, k)
        results[0] = 첫번째 쿼리의 top-k 문서의 인덱스 번호 (idx)
        scores[0] = 첫번째 쿼리의 BM25 점수
        '''
        results, scores = self._bm25.retrieve(
            tokenized_query,
            k=self._similarity_top_k
        )

        nodes_with_scores: List[NodeWithScore] = []  # 반드시 _retrieve는 List[NodeWithScore]이 반환타입이어야함
        for idx, score in zip(results[0], scores[0]):
            if score > 0:
                nodes_with_scores.append(
                    NodeWithScore(node=self._nodes[idx], score=float(score))
                )

        return nodes_with_scores

    def persist(self, path: str) -> None:
        self._bm25.save(path, corpus=None)
        config = {"similarity_top_k": self._similarity_top_k}
        with open(f"{path}/retriever.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def from_persist_dir(
        cls,
        path: str,
        nodes: List[BaseNode],
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "KiwiBM25Retriever":
        with open(f"{path}/retriever.json", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls.__new__(cls)
        instance._nodes = nodes
        instance._similarity_top_k = config["similarity_top_k"]

        if tokenizer is None:
            kiwi = Kiwi()
            instance._tokenizer = lambda text: [t.form for t in kiwi.tokenize(text)]
        else:
            instance._tokenizer = tokenizer

        instance._bm25 = bm25s.BM25.load(path, load_corpus=False)
        BaseRetriever.__init__(instance)
        return instance

tag_include=['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'MM', 'XR', 'SW', 'SL', 'SH', 'SN', 'SB']

corpus_tokenizer = partial(
    tokenize_kiwi,
    kiwi=kiwi,
    tag_include=tag_include,
    text_type="corpus",
    top_n=2,
    score_threshold=1.2,
)

query_tokenizer = partial(
    tokenize_kiwi,
    kiwi=kiwi,
    tag_include=tag_include,
    text_type="query",
    top_n=3,
    score_threshold=1.2,
)

kiwi_bm25_retriever = KiwiBM25Retriever(
    nodes=nodes,
    similarity_top_k=30,
    corpus_tokenizer=corpus_tokenizer,
    query_tokenizer=query_tokenizer
)

print("키위 세팅 끝")

print("퓨전 리트리버 시작")
fusion_retriever = QueryFusionRetriever(
    retrievers=[retriever, kiwi_bm25_retriever],
    similarity_top_k=30,  
    num_queries=1,
    use_async=False,mode="reciprocal_rerank"
)
print("퓨전 리트리버 끝")


train_set_dir = "/data/ephemeral/home/data/train_dataset"
train_dataset = load_from_disk(train_set_dir)

import tqdm

result_for_test = []

for i in tqdm.tqdm(range(len(train_dataset['train']['question']))):

    # 질문과 id
    test_q_query = train_dataset['train'][i]['question']
    test_q_id = train_dataset['train'][i]['id']

    # 골든리트리버 귀엽다
    retrieved_nodes_test = fusion_retriever.retrieve(test_q_query)


    # data for reranker
    docs_for_rerank_test = [n.node.text for n in retrieved_nodes_test]
    ids_for_rerank_test = [n.node.metadata['document_id'] for n in retrieved_nodes_test]


    # rerank result
    reranked_results_test = reranker.rerank(test_q_query, docs_for_rerank_test, ids_for_rerank_test, top_k=5)
    result_for_test.append([test_q_id, (list(np.array(reranked_results_test[1])[:,0].astype(int)))])
    
def convert_to_json(data):
    question_id = []
    document_list = []

    for q_id, doc_list in data:
        question_id.append(q_id)
        document_list.append(list(map(int, (doc_list))))
    result_dict = {
        "question_id": question_id,
        "document_id": document_list
    }
    return result_dict

json_test = convert_to_json(result_for_test)

file_path = '/data/ephemeral/home/pro-nlp-mrc-nlp-05/document/train_kiwi_hybrid.json'

with open(file_path, 'w') as f:
    json.dump(json_test, f)
