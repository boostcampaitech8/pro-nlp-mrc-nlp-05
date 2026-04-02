import os
import json
import torch
import tqdm
import re
import pickle
import networkx as nx
import numpy as np
from functools import partial
from collections import defaultdict
from typing import List, Dict, Callable, Optional, Tuple, Any, Union

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from kiwipiepy import Kiwi
from gliner import GLiNER
from sentence_transformers import CrossEncoder
import bm25s

# LlamaIndex 관련
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle, BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from datasets import load_from_disk

# --- 설정 상수 ---
GEMMA_MODEL_NAME = "google/gemma-3-4b-it"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
WIKI_DATA_PATH = './data/wikipedia_documents.json'
TRAIN_SET_DIR = "./data/test_dataset/"
KG_FILE_PATH = "./urimalsaem_graph_FINAL2.pkl" # 지식 그래프 경로
OUTPUT_FILE_PATH = './test_context_kiwi_NER_synonym_dense.json'

# --- 1. 환경 및 데이터 설정 ---
def setup_environment():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_wiki_data():
    with open(WIKI_DATA_PATH) as f:
        return json.load(f)

def load_train_dataset():
    return load_from_disk(TRAIN_SET_DIR)

# --- 2. 매핑 데이터 및 사전 정의 ---
KIWI_TO_URIMALSAEM_MAP = {
    "NNG": "명사", "NNP": "명사", "NNB": "의존 명사", "NR": "수사", "XR": "명사", "SN": "수사",
    "VV": "동사", "VA": "형용사", "MM": "관형사"
}

KIWI_TAGS = ['NNG', 'NNP', 'NNB', 'NR', 'VV', 'VA', 'MM', 'XR', 'SW', 'SL', 'SH', 'SN', 'SB']

GLINER_TAG = {
    "언어": {"CV_LANGUAGE": "문명_언어", "TR_HUMANITIES": "이론_철학/언어/역사"}, "문학": {"FD_HUMANITIES": "학문 분야_인문학", "AFA_DOCUMENT": "인공물_도서/서적 작품명"}, "역사": {"CV_CULTURE": "문명_문명/문화", "AF_CULTURAL_ASSET": "인공물_문화재", "DT_DYNASTY": "날짜_왕조시대", "DT_GEOAGE": "날짜_지질시대", "EV_WAR_REVOLUTION": "사건_전쟁/혁명", "EV_OTHERS": "사건_기타"}, "철학": {"TR_HUMANITIES": "이론_철학/언어/역사", "CV_TRIBE": "문명_민족/종족"}, "교육": {"OGG_EDUCATION": "기관_교육", "OGG_LIBRARY": "기관_도서관"}, "민속": {"CV_CULTURE": "문명_문명/문화", "EV_FESTIVAL": "사건_축제/영화제"}, "인문 일반": {"FD_HUMANITIES": "학문 분야_인문학"},
    "법률": {"CV_LAW": "문명_법/법률", "OGG_LAW": "기관_법률", "CV_POLICY": "문명_제도/정책"}, "군사": {"OGG_MILITARY": "기관_군사", "AF_WEAPON": "인공물_무기"}, "경영": {"OGG_ECONOMY": "기관_경제", "AFW_SERVICE_PRODUCTS": "인공물_서비스 상품"}, "경제": {"OGG_ECONOMY": "기관_경제", "CV_CURRENCY": "문명_통화", "CV_TAX": "문명_조세"}, "복지": {"CV_FUNDS": "문명_연금/기금"}, "정치": {"OGG_POLITICS": "기관_정부/공공", "CV_POLICY": "문명_제도/정책", "EV_ACTIVITY": "사건_사회운동/선언"}, "매체": {"OGG_MEDIA": "기관_미디어", "AFA_VIDEO": "인공물_영화/TV 프로그램"}, "행정": {"OGG_POLITICS": "기관_정부/공공"}, "심리": {"FD_SOCIAL_SCIENCE": "학문 분야_사회과학"}, "사회 일반": {"FD_SOCIAL_SCIENCE": "학문 분야_사회과학"},
    "지구": {"FD_SCIENCE": "학문 분야_과학", "MT_ROCK": "물질_암석"}, "지리": {"LC_OTHERS": "장소_기타", "LCG_MOUNTAIN": "장소_산/산맥", "LCG_RIVER": "장소_강/호수", "LCG_OCEAN": "장소_바다", "LCG_ISLAND": "장소_섬", "LCG_CONTINENT": "장소_대륙", "TM_DIRECTION": "용어_방향"}, "해양": {"LCG_OCEAN": "장소_바다", "LCG_BAY": "장소_반도/만"}, "천문": {"LC_SPACE": "장소_천체", "FD_SCIENCE": "학문 분야_과학"}, "환경": {"TM_CLIMATE": "용어_기후 지역", "FD_SCIENCE": "학문 분야_과학"}, "생명": {"TM_CELL_TISSUE_ORGAN": "용어_세포/조직/기관", "FD_SCIENCE": "학문 분야_과학"}, "동물": {"AM_INSECT": "동물_곤충", "AM_BIRD": "동물_조류", "AM_FISH": "동물_어류", "AM_MAMMALIA": "동물_포유류", "AM_AMPHIBIA": "동물_양서류", "AM_REPTILIA": "동물_파충류", "AM_TYPE": "동물_분류명", "AM_PART": "동물_부위명", "AM_OTHERS": "동물_기타"}, "식물": {"PT_FLOWER": "식물_꽃", "PT_GRASS": "식물_풀", "PT_TYPE": "식물_분류명", "PT_PART": "식물_부위명", "PT_OTHERS": "식물_기타", "PT_TREE": "식물_나무", "PT_FRUIT": "식물_과일/열매"}, "천연자원": {"MT_ELEMENT": "물질_원소"}, "수학": {"FD_SCIENCE": "학문 분야_과학", "TM_SHAPE": "용어_모양/형태", "QT_SIZE": "수량_넓이/면적", "QT_LENGTH": "수량_길이/거리", "QT_VOLUME": "수량_부피", "QT_PERCENTAGE": "수량_백분율"}, "물리": {"FD_SCIENCE": "학문 분야_과학", "TR_SCIENCE": "이론_과학", "QT_SPEED": "수량_속도", "QT_TEMPERATURE": "수량_온도", "QT_WEIGHT": "수량_무게"}, "화학": {"MT_CHEMICAL": "물질_화학", "MT_ELEMENT": "물질_원소", "MT_METAL": "물질_금속"}, "자연 일반": {"FD_SCIENCE": "학문 분야_과학"},
    "농업": {"PT_FRUIT": "식물_과일/열매", "FD_OTHERS": "학문 분야_기타"}, "수산업": {"AM_FISH": "동물_어류"}, "임업": {"PT_TREE": "식물_나무"}, "광업": {"MT_ROCK": "물질_암석", "MT_METAL": "물질_금속"}, "공업": {"AFW_OTHER_PRODUCTS": "인공물_기타 상품"}, "서비스업": {"AFW_SERVICE_PRODUCTS": "인공물_서비스 상품", "OGG_HOTEL": "기관_숙박 업체"}, "산업 일반": {"OGG_ECONOMY": "기관_경제"},
    "의학": {"FD_MEDICINE": "학문 분야_의학", "TR_MEDICINE": "이론_의학", "TMM_DISEASE": "용어_증상/질병"}, "약학": {"TMM_DRUG": "용어_약품"}, "한의": {"FD_MEDICINE": "학문 분야_의학"}, "수의": {"FD_MEDICINE": "학문 분야_의학"}, "식품": {"CV_FOOD": "문명_음식", "CV_DRINK": "문명_음료/술", "CV_FOOD_STYLE": "문명_음식 유형"}, "보건 일반": {"OGG_MEDICINE": "기관_의료"},
    "건설": {"AF_BUILDING": "인공물_건축물/토목건설물", "CV_BUILDING_TYPE": "문명_건축 양식"}, "교통": {"AF_TRANSPORT": "인공물_교통수단/운송수단", "AF_ROAD": "인공물_도로/철로"}, "기계": {"TMI_HW": "용어_IT 하드웨어"}, "전기·전자": {"TMI_HW": "용어_IT 하드웨어"}, "재료": {"MT_ELEMENT": "물질_원소"}, "정보·통신": {"TMI_SW": "용어_IT 소프트웨어", "TMI_HW": "용어_IT 하드웨어", "TMI_SITE": "용어_URL 주소", "TMI_EMAIL": "용어_이메일 주소", "TMI_MODEL": "용어_제품 모델명", "TMI_SERVICE": "용어_IT 서비스", "TMI_PROJECT": "용어_프로젝트"}, "공학 일반": {"FD_SCIENCE": "학문 분야_과학"},
    "체육": {"CV_SPORTS": "문명_스포츠", "OGG_SPORTS": "기관_스포츠", "CV_SPORTS_POSITION": "문명_스포츠 포지션", "CV_SPORTS_INST": "문명_스포츠 용품/도구", "EV_SPORTS": "사건_스포츠 행사", "TM_SPORTS": "용어_스포츠"}, "연기": {"AFA_PERFORMANCE": "인공물_춤/공연/연극 작품명"}, "영상": {"AFA_VIDEO": "인공물_영화/TV 프로그램"}, "무용": {"AFA_PERFORMANCE": "인공물_춤/공연/연극 작품명"}, "음악": {"AFA_MUSIC": "인공물_음악 작품명", "AF_MUSICAL_INSTRUMENT": "인공물_악기", "OGG_ART": "기관_예술"}, "미술": {"AFA_ART_CRAFT": "인공물_미술/조형 작품명", "FD_ART": "학문 분야_예술", "TM_COLOR": "용어_색깔"}, "복식": {"CV_CLOTHING": "문명_의복/섬유"}, "공예": {"AFA_ART_CRAFT": "인공물_미술/조형 작품명"}, "예체능 일반": {"FD_ART": "학문 분야_예술"},
    "가톨릭": {"OGG_RELIGION": "기관_종교"}, "기독교": {"OGG_RELIGION": "기관_종교"}, "불교": {"OGG_RELIGION": "기관_종교"}, "종교 일반": {"OGG_RELIGION": "기관_종교"},
    "인명": {"PS_NAME": "인물_사람", "PS_CHARACTER": "인물_가상 캐릭터", "CV_OCCUPATION": "문명_직업", "CV_POSITION": "문명_직위/직책", "CV_RELATION": "문명_가족/친족 관계"}, "지명": {"LCP_COUNTRY": "장소_국가", "LCP_PROVINCE": "장소_도/주 지역", "LCP_COUNTY": "장소_세부 행정구역", "LCP_CITY": "장소_도시", "LCP_CAPITALCITY": "장소_수도", "LC_OTHERS": "장소_기타"}, "책명": {"AFA_DOCUMENT": "인공물_도서/서적 작품명"}, "고유명 일반": {"OGG_OTHERS": "기관_기타"}
}

gliner_labels = []

for middle_cat, inner_map in GLINER_TAG.items():
    for gliner_code, leaf_label in inner_map.items():
        gliner_labels.append(leaf_label)
gliner_labels = sorted(list(set(gliner_labels)))

KIWI_TO_URIMALSAEM_MAP = {
    "NNG": "명사", "NNP": "명사", "NNB": "의존 명사", "NR": "수사", "XR": "명사", "SN": "수사",
    "VV": "동사", "VA": "형용사", "MM": "관형사",
    "SW": None, "SB": None, "SL": None, "SH": None
}

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
    chunk_size: int = 256,
    chunk_overlap: int = 128
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


def _fallback_tokenize(text: str) -> list[str]:
    """Kiwi 실패 시 단순 whitespace + 문자 기반 토큰화"""
    # 공백 분리 + 알파벳/숫자/기타 유니코드 단어 추출
    tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
    return [t for t in tokens]

def tokenize_kiwi(
    text: str,
    kiwi: Kiwi,
    tag_include: List[str],
    text_type: str,
    top_n: int,
    score_threshold: float = 1.2,
) -> list[str]:
    try:
        # 1. 토큰화할 텍스트가 문서(Corpus)일 때
        if text_type == "corpus":
            # 문서는 길 수 있으므로 top_n을 유동적으로 설정
            analyzed = kiwi.analyze(text, top_n=top_n + len(text) // 200)

            if not analyzed:
                return _fallback_tokenize(text)

            num_candi = 1
            # 1위 점수 기준 threshold 이내의 후보군 추가
            while (
                num_candi < len(analyzed)
                and analyzed[num_candi][1] > score_threshold * analyzed[0][1]
            ):
                num_candi += 1

        # 2. 토큰화할 텍스트가 쿼리(Query)일 때
        elif text_type == "query":
            analyzed = kiwi.analyze(text, top_n=top_n)

            if not analyzed:
                return _fallback_tokenize(text)

            num_candi = 3 # 쿼리는 좀 더 다양하게 후보를 봄

        # 3. 후보군에서 토큰 추출
        all_tokenized = [
            (t.form, t.tag)
            for nc in range(num_candi)
            for t in analyzed[nc][0]
        ]

        # 4. 중복 제거
        unique_tokenized = set(all_tokenized)

        # 5. [핵심 수정] 필터링은 하되, 태그(/NNG 등)는 떼고 '단어'만 리스트에 담음
        filtered = [
            form  # 🚨 수정됨: f"{form}/{tag}" -> form
            for form, tag in unique_tokenized
            if tag in tag_include
        ]

        return filtered if filtered else _fallback_tokenize(text)

    except Exception:
        return _fallback_tokenize(text)

class RichHybridTokenizer:
    def __init__(self, gliner_model: GLiNER, labels: List[str], kiwi_tags: List[str]):
        print("🔧 Rich Hybrid Tokenizer (Raw Mode: 필터링 없음) 초기화...")
        self.gliner = gliner_model
        self.labels = labels
        self.kiwi = Kiwi()

        # 사용자가 요청한 태그 리스트를 그대로 사용
        self.target_tags = set(kiwi_tags)

        # 🚨 C++의 '++' 등을 잡기 위해 SW(기호)가 target_tags에 없다면 강제로 추가 권장
        # (사용자님이 전달주실 kiwi_tags에 SW가 포함되어 있다고 가정하거나, 여기서 추가)
        # 붙임표도 일단 가져옴 (필요 없으면 나중에 매핑에서 None 처리)

    def tokenize(self, text: str) -> List[Dict[str, Any]]:
        if not text: return []

        token_info = {}

        # -------------------------------------------------------
        # 1. GLiNer: 핵심 개체명 & 카테고리 추출
        # -------------------------------------------------------
        try:
            preds = self.gliner.predict_entities(
                text, self.labels, flat_ner=True, threshold=0.1
            )
            for e in preds:
                raw_token = e['text'] # 공백 유지 (예: Visual Basic)
                category = e['label']

                if raw_token not in token_info:
                    token_info[raw_token] = {
                        'text': raw_token,
                        'category': category,
                        'pos': '(-)'
                    }
                else:
                    token_info[raw_token]['category'] = category

        except Exception as e:
            print(f"⚠️ GLiNer Error: {e}")

        # -------------------------------------------------------
        # 2. Kiwi: 형태소 분석 & 품사 태깅
        # -------------------------------------------------------
        try:
            res = self.kiwi.analyze(text, top_n=1)
            if res:
                for token in res[0][0]:
                    # 1. 타겟 태그인지 확인
                    if token.tag in self.target_tags:
                        word = token.form

                        # 🚨 [삭제됨] SW, SO 반복 문자 필터링 로직 제거!
                        # 이제 "++", "C#", "~~~~" 모두 있는 그대로 들어옵니다.

                        # 저장 로직
                        if word in token_info:
                            # GLiNer와 겹치면 POS 정보 업데이트
                            token_info[word]['pos'] = token.tag
                        else:
                            # Kiwi만 찾은 단어 추가
                            token_info[word] = {
                                'text': word,
                                'category': None,
                                'pos': token.tag
                            }
        except Exception as e:
            print(f"⚠️ Kiwi Error: {e}")

        return list(token_info.values())
    

from typing import List, Dict, Any
from collections import defaultdict
import json


def create_leaf_to_middle_map(simplified_map: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    """분류 체계를 평탄화하여 (상세 분류 -> 중분류 리스트) 맵을 생성"""
    leaf_to_middle = defaultdict(set)
    # GLINER_TAG 구조: {중분류: {코드: 상세분류}}
    for middle_cat, tag_dict in simplified_map.items():
        for gliner_code, leaf_value in tag_dict.items():
            leaf_to_middle[leaf_value].add(middle_cat)
    return {k: sorted(list(v)) for k, v in leaf_to_middle.items()}

def transform_tokens_enrich_data(token_list: List[Dict[str, Any]], leaf_to_mid_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    입력 토큰 리스트에 'dict_cat'(중분류)과 'dict_pos'(사전 품사)를 추가합니다.
    """
    enriched_list = []

    # 전역 변수 KIWI_TO_URIMALSAEM_MAP 사용 (윗 셀에서 정의됨)
    # 혹시 모르니 안전장치로 get 사용
    global KIWI_TO_URIMALSAEM_MAP

    for token in token_list:
        raw_text = token.get('text')
        leaf_cat = token.get('category')
        raw_pos = token.get('pos')

        # 1. 중분류 조회 (GLINER_TAG 기반 역추적)
        middle_categories = leaf_to_mid_map.get(leaf_cat, [])

        # 2. 품사 매핑 (KIWI_TO_URIMALSAEM_MAP 사용)
        dict_pos = KIWI_TO_URIMALSAEM_MAP.get(raw_pos)

        new_token = {
            'text': raw_text,
            'category': leaf_cat,
            'pos': raw_pos,
            'dict_cat': middle_categories, # 예: ['인명']
            'dict_pos': dict_pos           # 예: '명사'
        }
        enriched_list.append(new_token)

    return enriched_list

def load_knowledge_graph(file_path: str) -> nx.Graph:
    """pkl 파일에서 NetworkX 그래프를 로드합니다."""
    try:
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"✅ 그래프 로드 성공! (노드: {graph.number_of_nodes()}개, 엣지: {graph.number_of_edges()}개)")
        return graph
    except Exception as e:
        print(f"❌ 그래프 로드 실패: {e}")
        return None

class GraphRetriever:
    def __init__(self, graph: nx.Graph, min_weight: float = 0.7):
        print("🔍 GraphRetriever 초기화 (Word Indexing)...")
        self.graph = graph
        self.min_weight = min_weight
        self.word_index = defaultdict(list)
        self._build_word_index()

    def _build_word_index(self):
        for node_id, data in self.graph.nodes(data=True):
            word = data.get('word')
            if word:
                self.word_index[word].append(node_id)

    def retrieve(self, enriched_tokens: List[Dict[str, Any]]) -> Dict[str, float]:
        expanded_weights = defaultdict(float)
        original_texts = set()

        for item in enriched_tokens:
            raw_text = item['text']
            dict_cats = item['dict_cat']
            dict_pos = item['dict_pos']
            raw_pos = item['pos']

            # 1. 검색어 정규화
            search_text = raw_text
            if raw_pos in ['VV', 'VA']:
                search_text += '다'

            original_texts.add(raw_text)
            original_texts.add(search_text)

            # 2. 색인 조회
            candidate_ids = self.word_index.get(search_text, [])
            valid_ids = []

            # 3. 정밀 필터링 (수정된 로직) 🚨
            for nid in candidate_ids:
                node_cat = self.graph.nodes[nid].get('category')

                # Case A: GLiNER가 찾아준 중분류가 있으면 -> 강력 필터링 (유지)
                if dict_cats:
                    if node_cat in dict_cats:
                        valid_ids.append(nid)

                # Case B: 품사 정보만 있는 경우
                elif dict_pos:
                    # [수정 포인트] 동사/형용사는 엄격하게 검사
                    if dict_pos in ['동사', '형용사']:
                        if node_cat == dict_pos:
                            valid_ids.append(nid)

                    # [수정 포인트] 명사(NNG/NNP) 등은 카테고리 불일치 허용!
                    # "국가/명사" -> "국가/정치" (OK!)
                    else:
                        valid_ids.append(nid)

            # 4. 유의어 확장 (기존 동일)
            for nid in valid_ids:
                expanded_weights[search_text] = 1.0
                for neighbor, edge in self.graph[nid].items():
                    w = edge.get('weight', 0.0)
                    if w >= self.min_weight:
                        neighbor_word = self.graph.nodes[neighbor].get('word', neighbor)
                        if neighbor_word not in original_texts:
                            expanded_weights[neighbor_word] = max(expanded_weights[neighbor_word], w)

        return dict(expanded_weights)
    
class KiwiWeightedBM25Retriever(BaseRetriever):
    """
    [통합 버전]
    1. 문서는 Kiwi로 빠르게 인덱싱
    2. 질문은 RichTokenizer가 준 가중치(Dict)를 받아
    3. 내부에서 점수를 곱하고 더해서(Weight Sum) 결과를 반환
    """
    def __init__(
        self,
        nodes: List[BaseNode],
        similarity_top_k: int,
        corpus_tokenizer: Callable[[str], List[str]],         # 문서는 리스트 반환
        query_tokenizer: Callable[[str], Dict[str, float]]    # 🚨 질문은 딕셔너리 반환!
    ) -> None:
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self._corpus_tokenizer = corpus_tokenizer
        self._query_tokenizer = query_tokenizer

        print("🚀 [Index] 문서 인덱싱 시작 (Kiwi)...")
        # 문서는 기존대로 토큰 리스트로 변환하여 인덱싱
        corpus_tokens = [self._corpus_tokenizer(node.text) for node in nodes]

        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str

        # 1. 쿼리 토크나이징 (이제 딕셔너리를 받습니다!)
        # 예: {'시행착오': 1.0, '시오법': 0.7}
        weighted_query = self._query_tokenizer(query_str)

        # 2. [핵심 로직 이식] 가중치 기반 검색 (WeightedBM25S_Final 로직)
        doc_scores = defaultdict(float)

        # 단어 하나씩 검색해서 가중치 곱해서 더하기
        for token, weight in weighted_query.items():
            try:
                # bm25s는 입력을 이중 리스트로 받음 [[token]]
                results = self._bm25.retrieve([[token]], k=len(self._nodes))
            except Exception:
                continue

            if results.documents.size == 0:
                continue

            indices = results.documents[0]
            scores = results.scores[0]

            # (BM25점수 * 우리가 정한 가중치) 누적
            for idx, score in zip(indices, scores):
                doc_scores[idx] += (score * weight)

        # 3. 점수순 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 상위 k개만 잘라서 LlamaIndex 포맷(NodeWithScore)으로 변환
        top_k_docs = sorted_docs[:self._similarity_top_k]

        nodes_with_scores = []
        for idx, score in top_k_docs:
            nodes_with_scores.append(
                NodeWithScore(node=self._nodes[idx], score=float(score))
            )

        return nodes_with_scores

def tokenize_query_rich(
    text: str,
    tokenizer_obj: RichHybridTokenizer,
    l2m_map_obj: Dict[str, List[str]],
    graph_retriever_obj: GraphRetriever
) -> Dict[str, float]:
    """
    질문을 분석하여 개체명 정보를 추가하고, 지식 그래프를 통해 유의어로 확장된 가중치 BoW를 반환합니다.
    """
    if not text:
        return {}

    # 1. Rich Tokenizing (GLiNER + Kiwi)
    # 결과 예시: [{'text': '에케르트', 'category': '인물_사람', 'pos': 'NNP'}, ...]
    tokens = tokenizer_obj.tokenize(text)

    # 2. Enrich (메타데이터 및 사전 품사 추가)
    # 파일 내 정의된 transform_tokens_enrich_data 함수 호출
    enriched = transform_tokens_enrich_data(tokens, l2m_map_obj)

    # 3. Graph Expansion (유의어 찾기 및 가중치 계산)
    # 결과 예시: {'시행착오': 1.0, '시오법': 0.7}
    final_bow = graph_retriever_obj.retrieve(enriched)

    # 4. Safety Net (원본 단어 강제 포함)
    # 유의어 사전에 없는 단어라도 원본 질문의 키워드라면 가중치 1.0으로 추가
    for token in enriched:
        raw_text = token['text']
        if raw_text not in final_bow:
            final_bow[raw_text] = 1.0

    return final_bow

def setup_rich_query_tokenizer(
    gliner_model: GLiNER,
    synonym_graph: nx.Graph,
    labels: List[str],
    kiwi_tags: List[str],
    gliner_tag_map: Dict,
    min_weight: float = 0.7
) -> partial:
    """
    모든 검색 부품(모델, 그래프, 맵)을 조립하여 
    하나의 partial 토크나이저 함수를 반환합니다.
    """
    
    # 1. 기본 분석기 초기화
    tokenizer = RichHybridTokenizer(
        gliner_model=gliner_model,
        labels=labels,
        kiwi_tags=kiwi_tags
    )
    
    # 2. 메타데이터 매핑 생성
    l2m_map = create_leaf_to_middle_map(gliner_tag_map)
    
    # 3. 그래프 검색기 초기화
    graph_retriever = GraphRetriever(synonym_graph, min_weight=min_weight)
    
    # 4. Partial 함수 생성 (실제 리트리버에 주입될 함수)
    query_tokenizer_rich = partial(
        tokenize_query_rich,
        tokenizer_obj=tokenizer,
        l2m_map_obj=l2m_map,
        graph_retriever_obj=graph_retriever
    )
    
    return query_tokenizer_rich

def create_kiwi_synonym_retriever(
    nodes: List[BaseNode],
    rich_tokenizer: RichHybridTokenizer,
    l2m_map: Dict[str, List[str]],
    graph_retriever: GraphRetriever,
    kiwi_tags: List[str] = KIWI_TAGS,
    similarity_top_k: int = 30
) -> KiwiWeightedBM25Retriever:
    """
    Kiwi, GLiNER, 지식 그래프를 결합한 가중치 기반 BM25 리트리버를 생성합니다.
    """
    
    # 1. 쿼리용 토크나이저 조립 (질문 분석 -> 풍부화 -> 유의어 확장)
    # 🚨 Dict(단어:가중치)를 반환하는 tokenize_query_rich 함수를 partial로 래핑합니다.
    query_tokenizer_rich = partial(
        tokenize_query_rich,
        tokenizer_obj=rich_tokenizer,
        l2m_map_obj=l2m_map,
        graph_retriever_obj=graph_retriever
    )

    # 2. 문서용 토크나이저 조립 (기존 Kiwi - 단순 List[str] 반환)
    kiwi_instance = Kiwi()
    corpus_tokenizer = partial(
        tokenize_kiwi,
        kiwi=kiwi_instance,
        tag_include=kiwi_tags,
        text_type="corpus",
        top_n=2,
        score_threshold=1.2,
    )

    # 3. 통합 리트리버 생성 및 인덱싱 시작
    # 내부적으로 corpus_tokenizer를 사용하여 모든 노드를 인덱싱합니다.
    final_retriever = KiwiWeightedBM25Retriever(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        corpus_tokenizer=corpus_tokenizer,  # 문서는 리스트 방식
        query_tokenizer=query_tokenizer_rich # 질문은 딕셔너리(가중치) 방식
    )
    
    return final_retriever

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


def retrieve_formatted_results(
    final_retriever: QueryFusionRetriever,
    reranker: Reranker,
    train_dataset,
    output_path: str = OUTPUT_FILE_PATH,
    rerank_top_k: int = 5
) -> Dict:
    """
    Final Retriever와 Reranker를 사용하여 검색을 수행하고,
    결과를 JSON 파일로 저장합니다.
    """
    result_for_test = []

    for i in tqdm.tqdm(range(len(train_dataset['train']['question']))):
        # 질문과 id
        test_q_query = train_dataset['train'][i]['question']
        test_q_id = train_dataset['train'][i]['id']

        # final_retriever를 사용하여 검색 수행
        retrieved_nodes_test = final_retriever.retrieve(test_q_query)

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

def initialize_rich_pipeline():
    """
    Kiwi + GLiNER + 지식 그래프 유의어 확장이 포함된 
    전체 파이프라인 컴포넌트를 초기화하고 하이브리드 퓨전 리트리버를 반환합니다.
    """
    # 1. 환경 설정 및 데이터 로드
    setup_environment()
    wiki_data = load_wiki_data()
    id_to_title = get_id_to_title_mapping(wiki_data)
    train_dataset = load_train_dataset()
    
    # 2. 문서 및 노드(청크) 생성
    documents = create_documents_from_wiki(wiki_data)
    nodes = create_nodes_from_documents(documents)
    
    # 3. 모델 로드 (Embedding & LLM)
    embed_model = load_embedding_model()
    tokenizer_gemma, model_gemma = load_gemma()
    setup_llm_settings(model_gemma, tokenizer_gemma)
    
    # 4. GLiNER 모델 및 지식 그래프 로드
    gliner_model_obj = GLiNER.from_pretrained("lots-o/gliner-bi-ko-xlarge-v1").to("cuda")
    synonym_graph = load_knowledge_graph(KG_FILE_PATH)
    
    # 5. Rich 분석 부품 초기화
    rich_tokenizer = RichHybridTokenizer(
        gliner_model=gliner_model_obj, 
        labels=gliner_labels, 
        kiwi_tags=KIWI_TAGS
    )
    l2m_map = create_leaf_to_middle_map(GLINER_TAG)
    graph_retriever = GraphRetriever(synonym_graph, min_weight=0.7)
    
    # 6. 벡터 인덱스 및 Dense Retriever 생성 (추가됨)
    vector_index = create_faiss_vector_index(nodes, embed_model)
    dense_retriever = vector_index.as_retriever(similarity_top_k=50)
    
    # 7. 유의어 가중치 BM25 Retriever 생성 (Sparse)
    sparse_retriever = create_kiwi_synonym_retriever(
        nodes=nodes,
        rich_tokenizer=rich_tokenizer,
        l2m_map=l2m_map,
        graph_retriever=graph_retriever,
        kiwi_tags=KIWI_TAGS
    )
    
    # 8. 🚨 핵심 수정: 두 리트리버를 하나로 결합 (Fusion 단계)
    final_fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        similarity_top_k=30,          # 최종 후보군 개수
        num_queries=1,                # 쿼리 확장 미사용
        mode="reciprocal_rerank",      # RRF 방식 적용
        use_async=False
    )
    
    # 9. Reranker 생성
    reranker = Reranker()
    
    print("✅ 모든 파이프라인 컴포넌트(Synonym Hybrid Fusion) 초기화 완료!")

    return {
        'wiki_data': wiki_data,
        'id_to_title': id_to_title,
        'train_dataset': train_dataset,
        'documents': documents,
        'nodes': nodes,
        'embed_model': embed_model,
        'vector_index': vector_index,
        'final_retriever': final_fusion_retriever, # 결합된 리트리버 반환
        'reranker': reranker
    }

def main():
    """
    전체 하이브리드 검색 파이프라인을 가동합니다.
    1. 파이프라인 초기화
    2. 검색 및 리랭킹 수행
    3. 결과 저장
    """
    # 1. 모든 컴포넌트 초기화
    # initialize_rich_pipeline()에서 반환된 딕셔너리를 받습니다.
    components = initialize_rich_pipeline()
    
    # 2. 검색 수행 및 결과 JSON 저장
    # final_retriever를 사용하여 가중치 기반 유의어 확장이 적용된 검색을 수행합니다.
    json_result = retrieve_formatted_results(
        final_retriever=components['final_retriever'],
        reranker=components['reranker'],
        train_dataset=components['train_dataset'],
        output_path=OUTPUT_FILE_PATH,
        rerank_top_k=5
    )
    
    return json_result


if __name__ == "__main__":
    # 스크립트 실행 시 main 함수 호출
    print("Kiwi + NER + Synonym Hybrid 검색 파이프라인 시작")
    main()
    print("✨ 모든 작업이 성공적으로 완료되었습니다!")
