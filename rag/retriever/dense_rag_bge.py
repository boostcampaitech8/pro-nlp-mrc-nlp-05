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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import json
import tqdm
from dotenv import load_dotenv 

from Reranker import Reranker
from convert_to_json import convert_to_json

# --- 설정 상수 ---
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
WIKI_DATA_PATH = './data/wikipedia_documents.json'
TEST_SET_DIR = "./data/test_dataset/"
OUTPUT_FILE_PATH = './test_context_dense_bge.json'


def setup_environment():
    """환경 변수 로드 및 Hugging Face 로그인"""
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
        print("Hugging Face 로그인 성공!")
    else:
        print("에러: .env 파일에서 HF_TOKEN을 찾을 수 없습니다.")


def data_load(DIR):
    dataset = load_from_disk(DIR)
    return dataset

def wiki_load(WIKI_DATA_PATH):
    
    with open(WIKI_DATA_PATH) as f:
        wiki_data = json.load(f)
    documents: List[Document] = []
    for doc_id, data in wiki_data.items():
        # 'text' 필드를 문서 내용으로 사용
        data['text'] = data['text'].replace("\\n", "\n")
        data['text'] = data['text'].replace("\n", " ")
        
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

def chunking(documents, chunk_size, overlap):
    # 최종 청크 크기를 chunk_size=512
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    nodes: List[TextNode] = splitter.get_nodes_from_documents(documents)

    print(f"원본 문서 개수: {len(documents)}개")
    print(f"생성된 청크(Node) 개수: {len(nodes)}개")
    return nodes


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbedding:
    """임베딩 모델을 로드합니다."""
    return HuggingFaceEmbedding(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )    


def vectorize(embed_model, nodes):


    dummy_emb = embed_model.get_text_embedding("dim 체크용")
    dim = len(dummy_emb)
    faiss_index = faiss.IndexFlatIP(dim) 
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return vector_index

def initialize():
    setup_environment()
    test_dataset = data_load(TEST_SET_DIR)
    documents = wiki_load(WIKI_DATA_PATH)
    nodes = chunking(documents=documents, chunk_size=256, overlap=128)

    embedding_model = load_embedding_model()
    vector_index = vectorize(embedding_model, nodes)

    retriever = vector_index.as_retriever(similarity_top_k=60)

    reranker = Reranker()

    return {
        'test_dataset': test_dataset,
        'documents': documents,
        'nodes': nodes,
        'embed_model': embedding_model,
        'vector_index': vector_index,
        'dense_retriever': retriever,
        'reranker': reranker
    }

def retrieve_formatted_results(
    retriever,
    reranker,
    test_dataset,
    output_path: str = OUTPUT_FILE_PATH,
    rerank_top_k: int = 8
) -> Dict:



    result_for_validation = []

    for i in tqdm.tqdm(range(len(test_dataset['validation']['question']))):

        # 질문과 id
        validation_q_query = test_dataset['validation'][i]['question']
        validation_q_id = test_dataset['validation'][i]['id']

        # 골든리트리버 귀엽다
        retrieved_nodes_validation = retriever.retrieve(validation_q_query)


        # data for reranker
        docs_for_rerank_validation = [n.node.text for n in retrieved_nodes_validation]
        ids_for_rerank_validation = [n.node.metadata['document_id'] for n in retrieved_nodes_validation]



        # rerank result
        reranked_results_validation = reranker.rerank(validation_q_query, docs_for_rerank_validation, ids_for_rerank_validation, top_k=rerank_top_k)
        result = list(dict.fromkeys((list(map(int, ((list(np.array(reranked_results_validation[1])[:,0].astype(int)))))))))[:5]

        result_for_validation.append([validation_q_id, result])

        del validation_q_query, validation_q_id, retrieved_nodes_validation, docs_for_rerank_validation, ids_for_rerank_validation, reranked_results_validation, result
        if i%100 == 0:
            print(result_for_validation[i])

    json_test = convert_to_json(result_for_validation)
    file_path = output_path
    with open(file_path, 'w') as f:
        json.dump(json_test, f)


    return json_test

def main():
    # 파이프라인 초기화
    components = initialize()
    
    # 검색 수행 및 결과 JSON 저장
    json_result = retrieve_formatted_results(
        retriever=components['dense_retriever'],
        reranker=components['reranker'],
        test_dataset=components['test_dataset'],
        output_path=OUTPUT_FILE_PATH,
        rerank_top_k=5
    )

if __name__ == "__main__":
    main()