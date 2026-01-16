from datasets import load_from_disk
import os
from dotenv import load_dotenv
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
import json
from sentence_transformers import CrossEncoder 
from typing import List, Optional, Callable
import bm25s
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
from kiwipiepy import Kiwi
from functools import partial
import tqdm


# --- 설정 상수 ---
GEMMA_MODEL_NAME = "google/gemma-3-4b-it"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
WIKI_DATA_PATH = './data/wikipedia_documents.json'
TRAIN_SET_DIR = "./data/test_dataset/"
OUTPUT_FILE_PATH = './test_context_kiwi_dense.json'

# 포함할 품사 태그
TAG_INCLUDE = ['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'MM', 'XR', 'SW', 'SL', 'SH', 'SN', 'SB']


# 환경 설정 함수
def setup_environment():
    """환경 변수 로드 및 Hugging Face 로그인"""
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("Hugging Face 로그인 성공!")
    else:
        print("에러: .env 파일에서 HF_TOKEN을 찾을 수 없습니다.")


# 데이터 로드 함수
def load_wiki_data(wiki_path: str = WIKI_DATA_PATH) -> Dict:
    """Wikipedia 문서 데이터를 로드합니다."""
    with open(wiki_path) as f:
        wiki_data = json.load(f)
    return wiki_data


def get_id_to_title_mapping(wiki_data: Dict) -> Dict:
    """document_id와 title 매핑 딕셔너리를 생성합니다."""
    return {v["document_id"]: v["title"] for v in wiki_data.values()}


def load_train_dataset(train_set_dir: str = TRAIN_SET_DIR):
    """학습 데이터셋을 로드합니다."""
    return load_from_disk(train_set_dir)


# 모델 로드 함수
def load_gemma(model_name: str = GEMMA_MODEL_NAME):
    """Gemma 모델과 토크나이저를 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbedding:
    """임베딩 모델을 로드합니다."""
    return HuggingFaceEmbedding(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def setup_llm_settings(model, tokenizer):
    """LlamaIndex LLM 설정을 초기화합니다."""
    gemma_llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=8192,
    )
    Settings.llm = gemma_llm
    return gemma_llm


# 문서 처리 함수
def create_documents_from_wiki(wiki_data: Dict) -> List[Document]:
    """Wiki 데이터로부터 Document 객체 리스트를 생성합니다."""
    documents: List[Document] = []
    for doc_id, data in wiki_data.items():
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
    return documents


def create_nodes_from_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[TextNode]:
    """문서를 청킹하여 Node 리스트를 생성합니다."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: List[TextNode] = splitter.get_nodes_from_documents(documents)
    
    print(f"원본 문서 개수: {len(documents)}개")
    print(f"생성된 청크(Node) 개수: {len(nodes)}개")
    print(f"첫 번째 청크 텍스트 예시: {nodes[0].get_content()[:100]}...")
    
    return nodes


# 벡터 인덱스 생성 함수
def create_faiss_vector_index(
    nodes: List[TextNode],
    embed_model: HuggingFaceEmbedding
) -> VectorStoreIndex:
    """FAISS 기반 VectorStoreIndex를 생성합니다."""
    dummy_emb = embed_model.get_text_embedding("dim 체크용")
    dim = len(dummy_emb)
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("VectorStoreIndex 생성 시작")
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return vector_index


# Reranker 클래스
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
        scores = self.model.predict(pairs)
        scored_docs = list(zip(docs, scores))
        scored_id = list(zip(doc_id, scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        scored_id.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k], scored_id[:top_k]


# 토크나이저 관련 함수
def _fallback_tokenize(text: str) -> list[str]:
    """Kiwi 실패 시 단순 whitespace + 문자 기반 토큰화"""
    import re
    tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
    return [t for t in tokens]


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
            analyzed = kiwi.analyze(text, top_n=top_n + len(text) // 200)
            
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

        unique_tokenized = set(all_tokenized)

        filtered = [
            f"{form}/{tag}"
            for form, tag in unique_tokenized
            if tag in tag_include
        ]
        
        return filtered if filtered else _fallback_tokenize(text)
    
    except Exception:
        return _fallback_tokenize(text)


# BM25 Retriever 클래스
class KiwiBM25Retriever(BaseRetriever):
    """
    llamaIndex의 BaseRetriever를 상속받아 
    키위 토크나이저를 활용하는 커스텀 리트리버 클래스
    """

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


# Retriever 생성 함수
def create_kiwi_tokenizers(
    kiwi: Kiwi,
    tag_include: List[str] = TAG_INCLUDE
) -> tuple:
    """Kiwi 기반 corpus/query 토크나이저를 생성합니다."""
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
    
    return corpus_tokenizer, query_tokenizer


def create_fusion_retriever(
    vector_index: VectorStoreIndex,
    nodes: List[TextNode],
    corpus_tokenizer: Callable,
    query_tokenizer: Callable,
    vector_top_k: int = 50,
    bm25_top_k: int = 30,
    fusion_top_k: int = 30
) -> QueryFusionRetriever:
    """Vector Retriever와 BM25 Retriever를 결합한 Fusion Retriever를 생성합니다."""
    retriever = vector_index.as_retriever(similarity_top_k=vector_top_k)
    
    kiwi_bm25_retriever = KiwiBM25Retriever(
        nodes=nodes,
        similarity_top_k=bm25_top_k,
        corpus_tokenizer=corpus_tokenizer,
        query_tokenizer=query_tokenizer
    )
    
    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever, kiwi_bm25_retriever],
        similarity_top_k=fusion_top_k,
        num_queries=1,
        use_async=False,
        mode="reciprocal_rerank"
    )
    
    return fusion_retriever


# 결과 변환 함수
def convert_to_json(data: List) -> Dict:
    """결과 데이터를 JSON 형식으로 변환합니다."""
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


def save_results_to_json(data: Dict, file_path: str):
    """결과를 JSON 파일로 저장합니다."""
    with open(file_path, 'w') as f:
        json.dump(data, f)
    print(f"결과가 {file_path}에 저장되었습니다.")


# 메인 검색 함수
def retrieve_formatted_results(
    fusion_retriever: QueryFusionRetriever,
    reranker: Reranker,
    train_dataset,
    output_path: str = OUTPUT_FILE_PATH,
    rerank_top_k: int = 5
) -> Dict:
    """
    Fusion Retriever와 Reranker를 사용하여 검색을 수행하고,
    결과를 JSON 파일로 저장합니다.
    
    Args:
        fusion_retriever: QueryFusionRetriever 객체
        reranker: Reranker 객체
        train_dataset: 학습 데이터셋
        output_path: 결과 JSON 파일 저장 경로
        rerank_top_k: Reranking 후 반환할 상위 문서 개수
    
    Returns:
        Dict: {"question_id": [...], "document_id": [...]} 형식의 딕셔너리
    """
    result_for_test = []

    for i in tqdm.tqdm(range(len(train_dataset['train']['question']))):
        # 질문과 id
        test_q_query = train_dataset['train'][i]['question']
        test_q_id = train_dataset['train'][i]['id']

        retrieved_nodes_test = fusion_retriever.retrieve(test_q_query)

        # data for reranker
        docs_for_rerank_test = [n.node.text for n in retrieved_nodes_test]
        ids_for_rerank_test = [n.node.metadata['document_id'] for n in retrieved_nodes_test]

        # rerank result
        reranked_results_test = reranker.rerank(test_q_query, docs_for_rerank_test, ids_for_rerank_test, top_k=rerank_top_k)
        result_for_test.append([test_q_id, (list(np.array(reranked_results_test[1])[:,0].astype(int)))])
    
    json_result = convert_to_json(result_for_test)
    
    # JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(json_result, f)
    print(f"결과가 {output_path}에 저장되었습니다.")
    
    return json_result


# 파이프라인 초기화 함수
def initialize_pipeline():
    """전체 파이프라인을 초기화하고 필요한 컴포넌트들을 반환합니다."""
    # 환경 설정
    setup_environment()
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    # 데이터 로드
    wiki_data = load_wiki_data()
    id_to_title = get_id_to_title_mapping(wiki_data)
    train_dataset = load_train_dataset()
    
    # 문서 처리
    documents = create_documents_from_wiki(wiki_data)
    nodes = create_nodes_from_documents(documents)
    
    # 모델 로드
    embed_model = load_embedding_model()
    tokenizer, model = load_gemma()
    setup_llm_settings(model, tokenizer)
    
    # 인덱스 생성
    vector_index = create_faiss_vector_index(nodes, embed_model)
    
    # Retriever 생성
    kiwi = Kiwi()
    corpus_tokenizer, query_tokenizer = create_kiwi_tokenizers(kiwi)
    fusion_retriever = create_fusion_retriever(
        vector_index, nodes, corpus_tokenizer, query_tokenizer
    )
    
    # Reranker 생성
    reranker = Reranker()
    
    return {
        'wiki_data': wiki_data,
        'id_to_title': id_to_title,
        'train_dataset': train_dataset,
        'documents': documents,
        'nodes': nodes,
        'embed_model': embed_model,
        'vector_index': vector_index,
        'fusion_retriever': fusion_retriever,
        'reranker': reranker
    }


# 메인 실행
def main():
    # 파이프라인 초기화
    components = initialize_pipeline()
    
    # 검색 수행 및 결과 JSON 저장
    json_result = retrieve_formatted_results(
        fusion_retriever=components['fusion_retriever'],
        reranker=components['reranker'],
        train_dataset=components['train_dataset'],
        output_path=OUTPUT_FILE_PATH,
        rerank_top_k=5
    )
    
    return json_result


if __name__ == "__main__":
    main()