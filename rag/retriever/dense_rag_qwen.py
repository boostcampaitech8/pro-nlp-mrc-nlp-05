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

from convert_to_json import convert_to_json

# --- 설정 상수 ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-reranker-0.6B"
WIKI_DATA_PATH = './data/wikipedia_documents.json'
TEST_SET_DIR = "./data/test_dataset/"
OUTPUT_FILE_PATH = './test_context_dense_qwen.json'


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


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",  
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device).eval()

        # pad_token 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # yes / no 토큰 id
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        self.instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        self.max_length = 2048
        self.inner_max_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _build_inputs(self, pairs: List[str]):
        tok = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.inner_max_length,
        )
        # prefix/suffix 붙이기
        for i, ids in enumerate(tok["input_ids"]):
            tok["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens

        tok = self.tokenizer.pad(
            tok,
            padding=True,
            return_tensors="pt",
        )


        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        return tok

    @torch.no_grad()
    def _compute_scores(self, inputs) -> List[float]:
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :] 
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]

        stacked = torch.stack([false_logits, true_logits], dim=1)  
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().cpu().tolist() 

        return scores

    def rerank(self, query: str, docs: List[Dict], doc_id, top_k: int = 5):
        """
        query와 docs(문자열 리스트)를 받아, score 기준으로 다시 정렬해서 top_k만 반환합니다.
        """
        if not docs:
            return [], []


        pair_texts = [self._format_pair(query, d) for d in docs]

        inputs = self._build_inputs(pair_texts)
        scores = self._compute_scores(inputs)

        scored_docs = list(zip(docs, scores))
        scored_id = list(zip(doc_id, scores))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        scored_id.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k], scored_id[:top_k]


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