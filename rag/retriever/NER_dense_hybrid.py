from datasets import load_from_disk
import os
from dotenv import load_dotenv  # <--- 누락됨
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Callable, Optional, Tuple, Any, Union 
import numpy as np
import json
import tqdm
from functools import partial

# LlamaIndex 관련
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle, BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# 기타 라이브러리
import bm25s
from gliner import GLiNER
from sentence_transformers import CrossEncoder 

# --- 설정 상수 ---
GEMMA_MODEL_NAME = "google/gemma-3-4b-it"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
WIKI_DATA_PATH = './data/wikipedia_documents.json'
TRAIN_SET_DIR = "./data/test_dataset/"
OUTPUT_FILE_PATH = './test_context_NER_dense.json'

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
        scores = self.model.predict(pairs)  # shape (len(docs),)
        scored_docs = list(zip(docs, scores))
        scored_id = list(zip(doc_id, scores)) 

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        scored_id.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k], scored_id[:top_k]
    
# gliner 라벨 정의
entity_type_mapping = {
    "PS": { "PS_NAME": "인물_사람", "PS_CHARACTER": "인물_가상 캐릭터", "PS_PET": "인물_반려동물"},
    "FD": { "FD_SCIENCE": "학문 분야_과학", "FD_SOCIAL_SCIENCE": "학문 분야_사회과학", "FD_MEDICINE": "학문 분야_의학", "FD_ART": "학문 분야_예술", "FD_HUMANITIES": "학문 분야_인문학", "FD_OTHERS": "학문 분야_기타"},
    "TR": { "TR_SCIENCE": "이론_과학", "TR_SOCIAL_SCIENCE": "이론_사회과학", "TR_MEDICINE": "이론_의학", "TR_ART": "이론_예술", "TR_HUMANITIES": "이론_철학/언어/역사", "TR_OTHERS": "이론_기타"},
    "AF": { "AF_BUILDING": "인공물_건축물/토목건설물", "AF_CULTURAL_ASSET": "인공물_문화재", "AF_ROAD": "인공물_도로/철로", "AF_TRANSPORT": "인공물_교통수단/운송수단", "AF_MUSICAL_INSTRUMENT": "인공물_악기", "AF_WEAPON": "인공물_무기", "AFA_DOCUMENT": "인공물_도서/서적 작품명", "AFA_PERFORMANCE": "인공물_춤/공연/연극 작품명", "AFA_VIDEO": "인공물_영화/TV 프로그램", "AFA_ART_CRAFT": "인공물_미술/조형 작품명", "AFA_MUSIC": "인공물_음악 작품명", "AFW_SERVICE_PRODUCTS": "인공물_서비스 상품", "AFW_OTHER_PRODUCTS": "인공물_기타 상품"},
    "OG": { "OGG_ECONOMY": "기관_경제", "OGG_EDUCATION": "기관_교육", "OGG_MILITARY": "기관_군사", "OGG_MEDIA": "기관_미디어", "OGG_SPORTS": "기관_스포츠", "OGG_ART": "기관_예술", "OGG_MEDICINE": "기관_의료", "OGG_RELIGION": "기관_종교", "OGG_SCIENCE": "기관_과학", "OGG_LIBRARY": "기관_도서관", "OGG_LAW": "기관_법률", "OGG_POLITICS": "기관_정부/공공", "OGG_FOOD": "기관_음식 업체", "OGG_HOTEL": "기관_숙박 업체", "OGG_OTHERS": "기관_기타"},
    "LC": { "LCP_COUNTRY": "장소_국가", "LCP_PROVINCE": "장소_도/주 지역", "LCP_COUNTY": "장소_세부 행정구역", "LCP_CITY": "장소_도시", "LCP_CAPITALCITY": "장소_수도", "LCG_RIVER": "장소_강/호수", "LCG_OCEAN": "장소_바다", "LCG_BAY": "장소_반도/만", "LCG_MOUNTAIN": "장소_산/산맥", "LCG_ISLAND": "장소_섬", "LCG_CONTINENT": "장소_대륙", "LC_SPACE": "장소_천체", "LC_OTHERS": "장소_기타"},
    "CV": { "CV_CULTURE": "문명_문명/문화", "CV_TRIBE": "문명_민족/종족", "CV_LANGUAGE": "문명_언어", "CV_POLICY": "문명_제도/정책", "CV_LAW": "문명_법/법률", "CV_CURRENCY": "문명_통화", "CV_TAX": "문명_조세", "CV_FUNDS": "문명_연금/기금", "CV_ART": "문명_예술", "CV_SPORTS": "문명_스포츠", "CV_SPORTS_POSITION": "문명_스포츠 포지션", "CV_SPORTS_INST": "문명_스포츠 용품/도구", "CV_PRIZE": "문명_상/훈장", "CV_RELATION": "문명_가족/친족 관계", "CV_OCCUPATION": "문명_직업", "CV_POSITION": "문명_직위/직책", "CV_FOOD": "문명_음식", "CV_DRINK": "문명_음료/술", "CV_FOOD_STYLE": "문명_음식 유형", "CV_CLOTHING": "문명_의복/섬유", "CV_BUILDING_TYPE": "문명_건축 양식"},
    "DT": { "DT_DURATION": "날짜_기간", "DT_DAY": "날짜_일", "DT_WEEK": "날짜_주(주차)", "DT_MONTH": "날짜_달(월)", "DT_YEAR": "날짜_연(년)", "DT_SEASON": "날짜_계절", "DT_GEOAGE": "날짜_지질시대", "DT_DYNASTY": "날짜_왕조시대", "DT_OTHERS": "날짜_기타"},
    "TI": { "TI_DURATION": "시간_기간", "TI_HOUR": "시간_시각(시)", "TI_MINUTE": "시간_분", "TI_SECOND": "시간_초", "TI_OTHERS": "시간_기타"},
    "QT": { "QT_AGE": "수량_나이", "QT_SIZE": "수량_넓이/면적", "QT_LENGTH": "수량_길이/거리", "QT_COUNT": "수량_수량/빈도", "QT_MAN_COUNT": "수량_인원수", "QT_WEIGHT": "수량_무게", "QT_PERCENTAGE": "수량_백분율", "QT_SPEED": "수량_속도", "QT_TEMPERATURE": "수량_온도", "QT_VOLUME": "수량_부피", "QT_ORDER": "수량_순서", "QT_PRICE": "수량_금액", "QT_PHONE": "수량_전화번호", "QT_SPORTS": "수량_스포츠 수량", "QT_CHANNEL": "수량_채널 번호", "QT_ALBUM": "수량_앨범 수량", "QT_ADDRESS": "수량_주소 관련 숫자", "QT_OTHERS": "수량_기타 수량"},
    "EV": { "EV_ACTIVITY": "사건_사회운동/선언", "EV_WAR_REVOLUTION": "사건_전쟁/혁명", "EV_SPORTS": "사건_스포츠 행사", "EV_FESTIVAL": "사건_축제/영화제", "EV_OTHERS": "사건_기타"},
    "AM": { "AM_INSECT": "동물_곤충", "AM_BIRD": "동물_조류", "AM_FISH": "동물_어류", "AM_MAMMALIA": "동물_포유류", "AM_AMPHIBIA": "동물_양서류", "AM_REPTILIA": "동물_파충류", "AM_TYPE": "동물_분류명", "AM_PART": "동물_부위명", "AM_OTHERS": "동물_기타"},
    "PT": { "PT_FRUIT": "식물_과일/열매", "PT_FLOWER": "식물_꽃", "PT_TREE": "식물_나무", "PT_GRASS": "식물_풀", "PT_TYPE": "식물_분류명", "PT_PART": "식물_부위명", "PT_OTHERS": "식물_기타"},
    "MT": { "MT_ELEMENT": "물질_원소", "MT_METAL": "물질_금속", "MT_ROCK": "물질_암석", "MT_CHEMICAL": "물질_화학"},
    "TM": { "TM_COLOR": "용어_색깔", "TM_DIRECTION": "용어_방향", "TM_CLIMATE": "용어_기후 지역", "TM_SHAPE": "용어_모양/형태", "TM_CELL_TISSUE_ORGAN": "용어_세포/조직/기관", "TMM_DISEASE": "용어_증상/질병", "TMM_DRUG": "용어_약품", "TMI_HW": "용어_IT 하드웨어", "TMI_SW": "용어_IT 소프트웨어", "TMI_SITE": "용어_URL 주소", "TMI_EMAIL": "용어_이메일 주소", "TMI_MODEL": "용어_제품 모델명", "TMI_SERVICE": "용어_IT 서비스", "TMI_PROJECT": "용어_프로젝트", "TMIG_GENRE": "용어_게임 장르", "TM_SPORTS": "용어_스포츠"},
}
labels = []
for main_category in entity_type_mapping:
    sub_dict = entity_type_mapping[main_category]
    for key in sub_dict:
        labels.append(sub_dict[key])

# GLiNer XL 모델 로드
gliner_model = GLiNER.from_pretrained("lots-o/gliner-bi-ko-xlarge-v1")
gliner_model = gliner_model.to("cuda")


# gliner 기반 토크나이저
def tokenize_gliner_batch(
    texts: List[str],  # 🚨 입력이 단일 str이 아니라 List[str]입니다!
    model: GLiNER, 
    labels: List[str], 
    label_chunk_size: int = 20,
    score_threshold_ratio: float = 1.05
) -> List[List[str]]:
    if not texts: return []

    # 결과 저장용 (문서 개수만큼 빈 리스트 생성)
    batch_results = [[] for _ in texts]
    
    # 라벨 배치 처리 (Label Chunking)
    for i in range(0, len(labels), label_chunk_size):
        sub_labels = labels[i : i + label_chunk_size]
        try:
            # 🚀 model.batch_predict_entities 사용 (속도 향상의 핵심)
            batch_preds = model.batch_predict_entities(
                texts, sub_labels, flat_ner=True, threshold=0.1
            )
            # 결과 병합
            for doc_idx, entities in enumerate(batch_preds):
                batch_results[doc_idx].extend(entities)
        except Exception: continue
    
    # 후처리 (각 문서별로 점수 필터 & 문장 쪼개기 적용)
    final_batch_tokens = []
    
    for entities in batch_results:
        if not entities:
            final_batch_tokens.append([])
            continue
            
        # 1. 점수 필터링 (Relative Threshold)
        max_score = max(e['score'] for e in entities)
        cutoff_score = max_score / score_threshold_ratio
        filtered = [e for e in entities if e['score'] >= cutoff_score]
        
        # 2. 스마트 필터 (문장 쪼개기)
        doc_tokens = set()
        for e in filtered:
            token_text = e['text']
            # 띄어쓰기 2개 이상(3어절)이면 쪼개기
            if token_text.count(' ') >= 2:
                for t in token_text.split():
                    doc_tokens.add(t)
            else:
                doc_tokens.add(token_text)
        
        final_batch_tokens.append(list(doc_tokens))
        
    return final_batch_tokens

# gliner bm25s 리트리버
class GLiNerBM25Retriever(BaseRetriever):
    """partial 토크나이저를 받아서 배치 처리하는 Retriever"""
    def __init__(
        self,
        nodes: List[BaseNode],
        tokenizer: Callable[[List[str]], List[List[str]]], # 🚨 배치 토크나이저 시그니처
        similarity_top_k: int = 30,
        doc_batch_size: int = 64 # 🚀 문서 배치 사이즈 (한 번에 처리할 문서 수)
    ) -> None:
        
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self._tokenizer = tokenizer
        self.doc_batch_size = doc_batch_size

        print(f"🚀 GLiNer Batch Indexing... Docs: {len(nodes)}, Batch: {doc_batch_size}")
        
        corpus_tokens = []
        
        # 문서를 뭉텅이(Batch)로 잘라서 토크나이저 함수 호출
        for i in tqdm.tqdm(range(0, len(nodes), doc_batch_size), desc="Indexing"):
            batch_nodes = nodes[i : i + doc_batch_size]
            batch_texts = [n.text if n.text else "" for n in batch_nodes]
            
            # 여기서 partial로 만든 함수에 '리스트'를 던집니다!
            batch_tokens_list = self._tokenizer(batch_texts)
            corpus_tokens.extend(batch_tokens_list)

        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)
        
        print("✅ Indexing Complete!")
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        
        # 쿼리는 1개지만, 배치 함수니까 리스트로 감싸서 보냄
        # 결과도 리스트의 리스트니까 [0]으로 꺼냄
        query_tokens = self._tokenizer([query])[0]
        
        if not query_tokens: return []
        
        tokenized_query = [query_tokens]
        actual_k = min(self._similarity_top_k, len(self._nodes))
        if actual_k == 0: return []

        results, scores = self._bm25.retrieve(tokenized_query, k=actual_k)
        
        nodes_with_scores: List[NodeWithScore] = []
        for idx, score in zip(results[0], scores[0]):
            if score > 0:
                nodes_with_scores.append(
                    NodeWithScore(node=self._nodes[idx], score=float(score))
                )
        return nodes_with_scores


def create_gliner_tokenizers(
    model: GLiNER,  # GLiNER 모델 객체
    labels: List[str]
) -> partial:
    """
    GLiNER 모델과 라벨을 고정한 partial 토크나이저 튜플을 반환합니다.
    """
    
    # 1. Corpus용 토크나이저 (인덱싱용: 보통 더 엄격하거나 널널하게 조절)
    gliner_batch_tokenizer = partial(
        tokenize_gliner_batch,
        model=model,
        labels=labels,
        label_chunk_size=50,        # 라벨 20개씩 끊기
        score_threshold_ratio=2.5  # 점수 필터
)

    return gliner_batch_tokenizer


def create_gliner_fusion_retriever(
    vector_index: VectorStoreIndex,
    nodes: List[TextNode],
    gliner_tokenizer: Callable,
    vector_top_k: int = 50,
    bm25_top_k: int = 30,
    fusion_top_k: int = 30,
    doc_batch_size: int = 8
) -> QueryFusionRetriever:
    """
    Vector Retriever와 GLiNerBM25Retriever를 결합한 Fusion Retriever를 생성합니다.
    """
    # 1. 기본 Vector Retriever 생성
    vector_retriever = vector_index.as_retriever(similarity_top_k=vector_top_k)
    
    # 2. GLiNER 기반 BM25 Retriever 생성
    gliner_bm25_retriever = GLiNerBM25Retriever(
        nodes=nodes,
        similarity_top_k=bm25_top_k,
        tokenizer=gliner_tokenizer,
        doc_batch_size=doc_batch_size 
    )
    
    # 3. Reciprocal Rerank를 이용한 Fusion Retriever 구성
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, gliner_bm25_retriever],
        similarity_top_k=fusion_top_k,
        num_queries=1,           # 현재는 쿼리 확장 없이 1개만 사용
        use_async=False,         # 로컬 환경이나 디버깅 시 False 권장
        mode="reciprocal_rerank" # RRF 방식 적용
    )
    
    return fusion_retriever

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


def initialize_pipeline():
    """전체 파이프라인을 초기화하고 필요한 컴포넌트들을 반환합니다."""
    # 1~4. 환경 및 데이터 설정 (파일 내 정의된 함수들 사용)
    setup_environment()
    wiki_data = load_wiki_data()
    id_to_title = get_id_to_title_mapping(wiki_data)
    train_dataset = load_train_dataset()
    documents = create_documents_from_wiki(wiki_data)
    nodes = create_nodes_from_documents(documents)
    
    embed_model = load_embedding_model()
    tokenizer, model = load_gemma()
    setup_llm_settings(model, tokenizer)
    
    vector_index = create_faiss_vector_index(nodes, embed_model)

    # 5. GLiNER 토크나이저 생성 (파일 하단에 정의된 함수 호출)
    # 파일 내 전역 변수인 gliner_model과 labels를 인자로 사용합니다.
    gliner_tokenizer = create_gliner_tokenizers(
        model=gliner_model, 
        labels=labels
    )
    
    # 6. GLiNER Fusion Retriever 생성 (파일 하단에 정의된 함수 호출)
    fusion_retriever = create_gliner_fusion_retriever(
        vector_index=vector_index,
        nodes=nodes,
        gliner_tokenizer=gliner_tokenizer
    )
    
    # 7. Reranker 생성
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