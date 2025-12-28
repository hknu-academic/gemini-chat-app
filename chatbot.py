"""
============================================================
🎓 한경국립대학교 다전공 안내 AI챗봇
============================================================
버전: 3.1 (설정 파일 분리)
특징:
- Semantic Router로 의미 기반 의도 분류
- 설정 파일 분리 (config/*.yaml)
- 메시지, 매핑, 설정 외부화
============================================================

🔧 설치 필요 라이브러리:
pip install semantic-router sentence-transformers pyyaml

============================================================
"""

import streamlit as st
from google import genai
import pandas as pd
from streamlit_option_menu import option_menu 
from datetime import datetime
import os
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import re
import logging

# ============================================================
# 📌 설정 파일 로드
# ============================================================

def load_yaml_config(filename):
    """YAML 설정 파일 로드"""
    config_path = os.path.join('config', filename)
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# 설정 파일 로드
MESSAGES = load_yaml_config('messages.yaml')
MAPPINGS = load_yaml_config('mappings.yaml')
SETTINGS = load_yaml_config('settings.yaml')

# ============================================================
# 📌 설정에서 가져온 상수
# ============================================================

CONTACT_MESSAGE = "기본 메시지입니다." 

# 문의 메시지
CONTACT_MESSAGE = MESSAGES.get('contact', {}).get('default', 
    CONTACT_MESSAGE)

# 신청 기간 정보
APP_PERIOD = MESSAGES.get('application_period', {})
APP_PERIOD_TITLE = APP_PERIOD.get('title', "📅 다전공 신청 기간 안내")
APP_PERIOD_INTRO = APP_PERIOD.get('intro', "다전공 신청은 **매 학기 2회** 진행됩니다.")
APP_PERIOD_1ST = APP_PERIOD.get('first_semester', "전학기 **10월** / **12월**")
APP_PERIOD_2ND = APP_PERIOD.get('second_semester', "전학기 **4월** / **6월**")

# 링크
LINKS = MESSAGES.get('links', {})
ACADEMIC_NOTICE_URL = LINKS.get('academic_notice', "https://www.hknu.ac.kr/kor/562/subview.do")

# 에러 메시지
ERRORS = MESSAGES.get('errors', {})

# 경로
PATHS = SETTINGS.get('paths', {})
CURRICULUM_IMAGES_PATH = PATHS.get('curriculum_images', "images/curriculum")

# 앱 설정
APP_CONFIG = SETTINGS.get('app', {})
APP_TITLE = APP_CONFIG.get('title', "🎓 한경국립대 유연학사제도(다전공) 안내")

# 예시 질문
EXAMPLE_QUESTIONS = SETTINGS.get('example_questions', [
    "복수전공 신청 자격이 뭐야?",
    "신청 기간은 언제인가요?",
    "부전공이랑 복수전공 차이가 뭐야?",
    "마이크로디그리가 뭐야?"
])

# 타겟 제도
TARGET_PROGRAMS = SETTINGS.get('target_programs', ["복수전공", "부전공", "융합전공", "융합부전공"])

# 난이도 매핑
DIFFICULTY_STARS = MAPPINGS.get('difficulty_stars', {})

def convert_difficulty_to_stars(value):
    """숫자를 별점으로 변환"""
    if pd.isna(value) or value == '':
        return DIFFICULTY_STARS.get('default', '⭐⭐⭐')
    if isinstance(value, str) and '⭐' in value:
        return value
    try:
        num = int(float(value))
        return DIFFICULTY_STARS.get(num, DIFFICULTY_STARS.get('default', '⭐⭐⭐'))
    except:
        return DIFFICULTY_STARS.get('default', '⭐⭐⭐')

# Semantic Router 경고 메시지 숨기기
logging.getLogger("semantic_router").setLevel(logging.ERROR)

# === Semantic Router 설정 ===
SEMANTIC_ROUTER_ENABLED = True  # False로 변경하면 기존 키워드 방식으로 동작

# Semantic Router import (버전에 따라 경로가 다름)
SEMANTIC_ROUTER_AVAILABLE = False
Route = None
SemanticRouter = None  # 0.1.x에서는 RouteLayer 대신 SemanticRouter 사용
HuggingFaceEncoder = None
LocalIndex = None

try:
    # 0.1.x 버전 (최신)
    from semantic_router import Route
    from semantic_router.routers import SemanticRouter
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_router.index import LocalIndex
    SEMANTIC_ROUTER_AVAILABLE = True
    SEMANTIC_ROUTER_VERSION = "0.1.x"
except ImportError:
    try:
        # 0.0.x 버전 (구버전)
        from semantic_router import Route
        from semantic_router.layer import RouteLayer as SemanticRouter
        from semantic_router.encoders import HuggingFaceEncoder
        SEMANTIC_ROUTER_AVAILABLE = True
        SEMANTIC_ROUTER_VERSION = "0.0.x"
    except ImportError:
        SEMANTIC_ROUTER_AVAILABLE = False
        SEMANTIC_ROUTER_VERSION = None

if not SEMANTIC_ROUTER_AVAILABLE:
    st.warning("⚠️ Semantic Router가 설치되지 않았거나 호환되지 않습니다.\n키워드 기반 분류로 동작합니다.\n설치: pip install semantic-router sentence-transformers")

# === [AI 설정] Gemini API 연결 ===
GEMINI_API_KEY = "AIzaSyAyBEX3MRQv6q3RhNpznsfuDWKqhAlaGV8"
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# === 페이지 설정 ===
st.set_page_config(
    page_title="다전공 안내 AI챗봇",
    page_icon="🎓",
    layout="wide",
)

# === Streamlit 브랜딩 제거 및 모바일 최적화 ===
hide_streamlit_branding = """
<style>
footer {display: none !important;}
#MainMenu {visibility: hidden;}

/* 사이드바 토글 버튼은 유지 */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: block !important;
}

.stChatInputContainer {
    position: sticky;
    bottom: 0;
    background: white;
    padding: 0.75rem 0;
    z-index: 999;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}

.stChatMessage {
    margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        width: 85%;
    }
    .stChatInputContainer {
        padding: 0.5rem;
    }
    .stChatMessage {
        padding: 0.5rem !important;
    }
    .stButton button {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
}
</style>
"""
st.markdown(hide_streamlit_branding, unsafe_allow_html=True)


# === 자동 스크롤 함수 ===
def scroll_to_bottom():
    unique_id = str(uuid.uuid4())
    js = f"""
    <script>
        function scrollIntoView() {{
            var messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length > 0) {{
                var lastMessage = messages[messages.length - 1];
                lastMessage.scrollIntoView({{behavior: "smooth", block: "end"}});
            }}
        }}
        setTimeout(scrollIntoView, 300);
        setTimeout(scrollIntoView, 500);
    </script>
    """
    st.components.v1.html(js, height=0)


# === 세션 상태 초기화 ===
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'page' not in st.session_state:
        st.session_state.page = "AI챗봇 상담"


# ============================================================
# 📂 데이터 로드
# ============================================================

@st.cache_data
def load_excel_data(file_path, sheet_name=0):
    """엑셀 파일 로드 (기본: 첫 번째 시트)"""
    try:
        if os.path.exists(file_path):
            result = pd.read_excel(file_path, sheet_name=sheet_name)
            # sheet_name=None인 경우 dict 반환되므로 처리
            if isinstance(result, dict):
                # 첫 번째 시트 반환
                first_sheet = list(result.values())[0] if result else pd.DataFrame()
                return first_sheet
            return result
        return pd.DataFrame()
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return pd.DataFrame()


@st.cache_data
def load_program_info():
    df = load_excel_data('data/programs.xlsx')
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    programs = {}
    for _, row in df.iterrows():
        name = row.get('제도명', '')
        if name and pd.notna(name):
            # NaN 값 처리 함수
            def safe_get(key, default=''):
                val = row.get(key, default)
                return default if pd.isna(val) else val
            
            programs[name] = {
                'description': safe_get('설명', ''),
                'qualification': safe_get('신청자격', ''),
                'credits_general': safe_get('이수학점(교양)', ''),
                'credits_primary': safe_get('원전공 이수학점', ''),
                'credits_multi': safe_get('다전공 이수학점', ''),
                'degree': safe_get('학위기 표기', '-'),
                'features': str(safe_get('특징', '')).split('\n') if safe_get('특징', '') else [],
                'notes': safe_get('기타', ''),
                'difficulty': convert_difficulty_to_stars(safe_get('난이도', '3')),
                'graduation_certification': safe_get('졸업인증', '-'),
                'graduation_exam': safe_get('졸업시험', '-'),
            }
    return programs


@st.cache_data
def load_curriculum_mapping():
    try:
        if os.path.exists('data/curriculum_mapping.xlsx'):
            return pd.read_excel('data/curriculum_mapping.xlsx')
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])
    except:
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])


@st.cache_data
def load_courses_data():
    try:
        if os.path.exists('data/courses.xlsx'):
            return pd.read_excel('data/courses.xlsx')
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])
    except:
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])


@st.cache_data
def load_faq_data():
    df = load_excel_data('data/faq.xlsx')
    if df.empty:
        return []
    return df.to_dict('records')


@st.cache_data
def load_majors_info():
    return load_excel_data('data/majors_info.xlsx')


@st.cache_data
def load_graduation_requirements():
    return load_excel_data('data/graduation_requirements.xlsx')


@st.cache_data
def load_primary_requirements():
    return load_excel_data('data/primary_requirements.xlsx')


# 데이터 로드
PROGRAM_INFO = load_program_info()
CURRICULUM_MAPPING = load_curriculum_mapping()
COURSES_DATA = load_courses_data()
FAQ_DATA = load_faq_data()
MAJORS_INFO = load_majors_info()
GRADUATION_REQ = load_graduation_requirements()
PRIMARY_REQ = load_primary_requirements()

# 전체 데이터 딕셔너리
ALL_DATA = {
    'programs': PROGRAM_INFO,
    'curriculum': CURRICULUM_MAPPING,
    'courses': COURSES_DATA,
    'faq': FAQ_DATA,
    'majors': MAJORS_INFO,
    'grad_req': GRADUATION_REQ,
    'primary_req': PRIMARY_REQ,
}


# ============================================================
# 🧠 Semantic Router 설정 (Level 2 핵심!)
# ============================================================

# === 의도별 예시 문장 (Semantic Router용) ===
INTENT_UTTERANCES = {
    'QUALIFICATION': [
        "신청 자격이 어떻게 되나요?",
        "지원 자격 알려주세요",
        "누가 신청할 수 있어요?",
        "자격 요건이 뭐예요?",
        "나도 신청 가능해?",
        "몇 학년부터 할 수 있어요?",
        "2학년인데 가능한가요?",
        "학점이 낮아도 되나요?",
        "조건이 어떻게 돼?",
        "신청 조건 알려줘",
        "자격이 되는지 모르겠어",
        "이거 해도 돼?",
        "나 자격 있어?",
        "신청 자격 조건",
        "지원 가능 여부",
    ],
    
    'APPLICATION_PERIOD': [
        "신청 기간이 언제예요?",
        "언제 신청해요?",
        "마감일이 언제야?",
        "지원 기간 알려주세요",
        "언제까지 신청할 수 있어요?",
        "접수 기간이 어떻게 돼?",
        "몇 월에 신청해?",
        "신청 시작일이 언제야?",
        "기간이 얼마나 남았어?",
        "지금 신청 가능해?",
        "이번 학기 신청 기간",
        "다음 학기 신청은 언제?",
        "신청 일정 알려줘",
        "접수 마감일",
        "언제부터 언제까지야?",
    ],
    
    'APPLICATION_METHOD': [
        "신청 방법이 어떻게 되나요?",
        "어떻게 신청해요?",
        "신청 절차 알려주세요",
        "지원하려면 어떻게 해야 해?",
        "신청하는 법 알려줘",
        "어디서 신청해?",
        "온라인으로 신청 가능해?",
        "신청서 어디서 받아?",
        "절차가 어떻게 돼?",
        "지원 방법이 뭐야?",
        "신청하고 싶은데 어떻게 해?",
        "접수 방법",
        "신청 프로세스",
        "지원 절차 설명해줘",
        "어디로 가야해?",
    ],
    
    'CANCEL': [
        "포기하고 싶어요",
        "취소 방법 알려주세요",
        "철회하려면 어떻게 해?",
        "그만두고 싶어",
        "중단하고 싶은데",
        "포기 신청 어떻게 해?",
        "취소할 수 있어?",
        "포기 기간이 언제야?",
        "취소 가능한가요?",
        "다전공 포기",
        "복수전공 취소",
        "포기하면 어떻게 돼?",
        "취소 절차",
        "포기 방법",
        "안 하고 싶어",
    ],
    
    'CHANGE': [
        "변경하고 싶어요",
        "전공 바꾸고 싶어",
        "수정할 수 있나요?",
        "전환하려면 어떻게 해?",
        "다른 전공으로 변경",
        "복수전공에서 부전공으로 바꾸고 싶어",
        "변경 가능한가요?",
        "전공 변경 방법",
        "수정 절차",
        "바꿀 수 있어?",
        "변경 신청",
        "전환 방법",
        "다른 걸로 바꾸고 싶어",
    ],
    
    'PROGRAM_COMPARISON': [
        "복수전공이랑 부전공 차이가 뭐야?",
        "뭐가 다른 거야?",
        "차이점 알려줘",
        "비교해줘",
        "뭐가 더 좋아?",
        "어떤 게 나을까?",
        "융합전공이랑 복수전공 비교",
        "둘 다 하면 어떻게 돼?",
        "차이점이 뭐예요?",
        "비교해서 설명해줘",
        "뭐가 유리해?",
        "둘 중에 뭐가 좋아?",
        "장단점 비교",
    ],
    
    'CREDIT_INFO': [
        "학점이 몇 학점이야?",
        "이수 학점 알려줘",
        "졸업하려면 몇 학점 필요해?",
        "본전공 학점이 줄어들어?",
        "학점 변화 알려줘",
        "총 학점이 어떻게 돼?",
        "전필 몇 학점이야?",
        "전선 학점은?",
        "교양 학점은 어떻게 돼?",
        "학점 요건",
        "졸업 요건 학점",
        "필요한 학점 수",
        "이수해야 하는 학점",
    ],
    
    'PROGRAM_INFO': [
        "복수전공이 뭐야?",
        "부전공이 뭔가요?",
        "융합전공 설명해줘",
        "마이크로디그리가 뭐예요?",
        "연계전공이 뭐지?",
        "이게 뭐야?",
        "알려줘",
        "설명해줘",
        "무슨 제도야?",
        "어떤 건가요?",
        "소단위전공이 뭐야?",
        "융합부전공 설명",
        "제도 설명해줘",
    ],
    
    'COURSE_SEARCH': [
        "어떤 과목 들어야 해?",
        "커리큘럼 알려줘",
        "수업 뭐 들어?",
        "과목 리스트 보여줘",
        "뭐 배워?",
        "교과목 알려줘",
        "강의 뭐 있어?",
        "필수 과목이 뭐야?",
        "선택 과목은?",
        "이수 과목 목록",
        "과목 추천해줘",
        "어떤 강의 들어야 해?",
    ],
    
    'CONTACT_SEARCH': [
        "연락처 알려줘",
        "전화번호가 뭐야?",
        "문의 어디로 해?",
        "사무실 어디야?",
        "담당자 연락처",
        "어디로 전화해?",
        "문의처 알려줘",
        "홈페이지 주소",
        "위치가 어디야?",
        "연락할 곳",
    ],
    
    'RECOMMENDATION': [
        "뭐가 좋을까?",
        "추천해줘",
        "어떤 게 좋아?",
        "나한테 맞는 거 뭐야?",
        "뭐 해야 할까?",
        "고민이야 뭐 할지",
        "어떤 걸 선택해야 할까?",
        "추천 좀 해줘",
        "나한테 어떤 게 맞아?",
        "뭐가 유리할까?",
        "골라줘",
        "선택 도와줘",
        "뭐 하면 좋을까?",
        "조언 좀 해줘",
    ],
    
    'GREETING': [
        "안녕",
        "안녕하세요",
        "하이",
        "hello",
        "hi",
        "반가워",
        "처음이야",
        "시작",
        "안녕!",
        "헬로",
    ],
    
    # 🚫 범위 외 질문 (다전공과 무관한 질문)
    'OUT_OF_SCOPE': [
        "오늘 날씨 어때?",
        "맛집 추천해줘",
        "영화 추천해줘",
        "게임 추천해줘",
        "연애 상담 해줘",
        "취업 어떻게 해?",
        "공모전 추천해줘",
        "동아리 추천해줘",
        "기숙사 신청 어떻게 해?",
        "장학금 어떻게 받아?",
        "학식 메뉴 뭐야?",
        "도서관 몇시까지 해?",
        "셔틀버스 시간표 알려줘",
        "수강신청 어떻게 해?",
        "성적 정정 방법",
        "휴학 신청 방법",
        "졸업 요건 뭐야?",
        "교환학생 어떻게 가?",
        "인턴 어떻게 구해?",
        "자기소개서 써줘",
        "이력서 봐줘",
        "코딩 알려줘",
        "파이썬 가르쳐줘",
        "수학 문제 풀어줘",
        "영어 번역해줘",
        "과제 해줘",
        "레포트 써줘",
        "너 누구야?",
        "AI야?",
        "사람이야?",
    ],
    
    # 🚫 욕설/비속어 차단
    'BLOCKED': [
        "시발", "씨발", "ㅅㅂ", "ㅆㅂ", "씨빨", "시빨",
        "병신", "ㅂㅅ", "병딱", "븅신",
        "지랄", "ㅈㄹ", "지럴",
        "개새끼", "개색끼", "개세끼", "ㄱㅅㄲ",
        "꺼져", "닥쳐", "죽어", "뒤져",
        "미친", "미쳤", "ㅁㅊ", "미친놈", "미친년",
        "씹", "ㅆ", "씹새", "씹놈",
        "존나", "졸라", "ㅈㄴ",
        "애미", "애비", "엠창", "앰창",
        "좆", "ㅈ같", "좃",
        "걸레", "창녀", "보지", "자지",
        "fuck", "shit", "damn", "bitch",
        "썅", "엿먹어", "엿이나", "좇까",
    ],
}

# === 기존 키워드 (폴백용) ===
INTENT_KEYWORDS = {
    'QUALIFICATION': [
        '신청자격', '지원자격', '자격요건', '신청요건', '자격조건',
        '자격이어떻게', '자격은', '누가신청', '신청할수있', '지원할수있',
        '자격이뭐', '자격알려', '자격요건이', '신청자격이', '자격이어떻게돼',
        '자격어떻게', '누가할수있', '신청조건', '지원조건', '조건이뭐',
        '자격조건이', '신청가능', '지원가능'
    ],
    'APPLICATION_PERIOD': [
        '신청기간', '지원기간', '접수기간', '언제신청', '언제지원',
        '신청은언제', '지원은언제', '신청언제', '기간이언제', '기간알려',
        '마감일', '시작일', '종료일', '접수일', '신청일', '언제까지',
        '기간이어떻게', '몇월', '언제부터', '언제해'
    ],
    'APPLICATION_METHOD': [
        '신청방법', '지원방법', '신청절차', '지원절차', '어떻게신청',
        '어떻게지원', '신청어떻게', '절차가어떻게', '방법알려',
        '신청하는법', '지원하는법', '신청하려면', '지원하려면',
        '어디서신청', '어디서지원', '절차알려', '방법이뭐'
    ],
    'CANCEL': [
        '포기', '취소', '철회', '그만', '중단', '취소방법', '포기방법',
        '취소하려면', '포기하려면', '취소할수있', '포기할수있',
        '취소언제', '포기언제', '취소기간', '포기기간'
    ],
    'CHANGE': [
        '변경', '수정', '바꾸', '전환', '변경방법', '변경하려면',
        '바꾸려면', '전환하려면', '변경할수있', '바꿀수있'
    ],
    'PROGRAM_COMPARISON': [
        '차이', '비교', 'vs', '다른점', '뭐가달라', '어떻게달라',
        '무슨차이', '뭐가다른', '차이점', '비교해줘', '뭐가좋'
    ],
    'CREDIT_INFO': [
        '학점', '이수학점', '졸업요건', '필요한학점', '몇학점', 
        '졸업학점', '학점이', '변해', '줄어', '늘어', '학점변화',
        '본전공학점', '다전공학점', '총학점'
    ],
    'PROGRAM_INFO': [
        '뭐야', '무엇', '뭔가요', '뭐에요', '알려줘', '설명',
        '무엇인가', '이뭐야', '가뭐야', '은뭐', '는뭐'
    ],
    'COURSE_SEARCH': [
        '과목', '수업', '강의', '커리큘럼', '교육과정', '이수과목',
        '뭐배워', '뭐듣', '과목리스트', '과목알려', '강의알려',
        '교과목', '과목추천'
    ],
    'CONTACT_SEARCH': [
        '연락처', '전화번호', '문의', '번호', '사무실',
        '어디있', '위치', '전화', '홈페이지', '사이트'
    ],
    'RECOMMENDATION': [
        '추천', '뭐할까', '선택', '고민', '좋을까', '어떤게좋',
        '추천해줘', '골라줘', '뭐가좋아', '어떤걸', '뭐해야'
    ],
    'GREETING': [
        '안녕', '하이', 'hello', 'hi', '반가', '처음', '시작'
    ],
    # 🚫 범위 외 질문
    'OUT_OF_SCOPE': [
        '날씨', '맛집', '영화', '게임', '연애', '취업', '공모전', '동아리',
        '기숙사', '장학금', '학식', '도서관', '셔틀', '버스', '수강신청',
        '성적정정', '휴학', '교환학생', '인턴', '자소서', '이력서',
        '코딩', '파이썬', '수학', '영어', '번역', '과제', '레포트',
        '너누구', '사람이야', 'AI야', '뭐야너', '정체가뭐'
    ],
    # 🚫 욕설/비속어 차단
    'BLOCKED': [
        '시발', '씨발', 'ㅅㅂ', 'ㅆㅂ', '병신', 'ㅂㅅ', '지랄', 'ㅈㄹ',
        '개새끼', 'ㄱㅅㄲ', '꺼져', '닥쳐', '죽어', '뒤져', '미친', 'ㅁㅊ',
        '씹', '존나', 'ㅈㄴ', '애미', '애비', '좆', '걸레', '창녀',
        'fuck', 'shit', 'bitch', '썅', '엿먹어'
    ],
}

# === 제도 키워드 (비교/설명용) ===
PROGRAM_KEYWORDS = {
    '복수전공': ['복수전공', '복전', '복수'],
    '부전공': ['부전공', '부전'],
    '융합전공': ['융합전공', '융합'],
    '융합부전공': ['융합부전공'],
    '연계전공': ['연계전공', '연계'],
    '마이크로디그리': ['마이크로디그리', '마이크로', 'md', '소단위전공과정', '소단위전공', '소단위', '마디'],
}


# === Semantic Router 초기화 (캐싱) ===
@st.cache_resource
def initialize_semantic_router():
    """Semantic Router 초기화 (한 번만 실행)"""
    if not SEMANTIC_ROUTER_AVAILABLE or not SEMANTIC_ROUTER_ENABLED:
        return None
    
    # 필수 클래스가 import 되었는지 확인
    if Route is None or SemanticRouter is None or HuggingFaceEncoder is None:
        return None
    
    try:
        # 한국어 임베딩 모델 (무료)
        encoder = HuggingFaceEncoder(name="jhgan/ko-sroberta-multitask")
        
        # Route 생성
        routes = []
        for intent_name, utterances in INTENT_UTTERANCES.items():
            route = Route(
                name=intent_name,
                utterances=utterances,
            )
            routes.append(route)
        
        # SemanticRouter 생성 (0.1.x 버전) - LocalIndex 명시적 지정
        if LocalIndex is not None:
            index = LocalIndex()
            router = SemanticRouter(encoder=encoder, routes=routes, index=index)
        else:
            router = SemanticRouter(encoder=encoder, routes=routes)
        
        return router
    
    except Exception as e:
        st.warning(f"⚠️ Semantic Router 초기화 실패: {e}\n키워드 기반 분류로 동작합니다.")
        return None


# Semantic Router 인스턴스
SEMANTIC_ROUTER = initialize_semantic_router()


# === AI 의도 분류용 프롬프트 ===
INTENT_CLASSIFICATION_PROMPT = """당신은 질문 분류 AI입니다. 아래 의도 중 가장 적합한 하나를 선택하세요.

[의도 목록]
1. QUALIFICATION - 신청 자격, 지원 자격, 누가 신청 가능한지
2. APPLICATION_PERIOD - 신청 기간, 언제 신청, 마감일
3. APPLICATION_METHOD - 신청 방법, 절차, 어떻게 신청
4. CANCEL - 포기, 취소, 철회
5. CHANGE - 변경, 수정, 전환
6. PROGRAM_COMPARISON - 제도 비교, 차이점 (복수전공 vs 부전공 등)
7. PROGRAM_INFO - 특정 제도 설명 (복수전공이 뭐야?)
8. CREDIT_INFO - 학점, 이수 학점, 졸업 요건
9. COURSE_SEARCH - 과목 조회, 커리큘럼, 수업
10. CONTACT_SEARCH - 연락처, 전화번호, 사무실
11. RECOMMENDATION - 추천, 어떤 게 좋을까, 선택 고민
12. GREETING - 인사 (안녕, 하이)
13. OUT_OF_SCOPE - 다전공/유연학사제도와 전혀 무관한 질문 (날씨, 맛집, 취업, 휴학, 장학금, 수강신청, 기숙사 등)

[규칙]
- 반드시 의도 이름만 출력 (예: QUALIFICATION)
- 여러 의도가 섞여 있으면 가장 핵심적인 것 선택
- 다전공/복수전공/부전공/융합전공/마이크로디그리/연계전공과 관련없는 질문은 OUT_OF_SCOPE
"""


# ============================================================
# 🔥 의도 분류 함수 (Semantic Router 적용!)
# ============================================================

def extract_programs(text):
    """텍스트에서 제도명 추출"""
    found = []
    text_lower = text.lower()
    for program, keywords in PROGRAM_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                if program not in found:
                    found.append(program)
                break
    return found


def extract_additional_info(user_input, intent):
    """추가 정보 추출"""
    info = {}
    user_clean = user_input.lower().replace(' ', '')
    
    # 제도명 추출
    found_programs = extract_programs(user_clean)
    if found_programs:
        info['programs'] = found_programs
        info['program'] = found_programs[0]
    
    # 학번 추출
    year_match = re.search(r'(20\d{2})', user_input)
    if year_match:
        info['year'] = int(year_match.group(1))
    
    # 학점 추출
    credit_match = re.search(r'(\d+)\s*학점', user_input)
    if credit_match:
        info['credits'] = int(credit_match.group(1))
    
    # 전공명 추출 (COURSE_SEARCH, CONTACT_SEARCH 등에 필요)
    major_patterns = [
        r'([가-힣A-Za-z]+(?:융합)?전공)',  # ~전공
        r'([가-힣A-Za-z]+학과)',  # ~학과
    ]
    
    for pattern in major_patterns:
        major_match = re.search(pattern, user_input)
        if major_match:
            major_name = major_match.group(1)
            # 제도명은 제외 (복수전공, 부전공 등)
            if major_name not in ['복수전공', '부전공', '융합전공', '융합부전공', '연계전공', '다전공']:
                info['major'] = major_name
                break
    
    return info


def classify_with_semantic_router(user_input):
    """Semantic Router를 사용한 의도 분류"""
    if SEMANTIC_ROUTER is None:
        return None, 0.0
    
    try:
        result = SEMANTIC_ROUTER(user_input)
        if result and result.name:
            # score는 result에서 가져올 수 없으므로 기본값 사용
            return result.name, 0.8
        return None, 0.0
    except Exception as e:
        return None, 0.0


def classify_with_keywords(user_input):
    """키워드 기반 의도 분류 (폴백)"""
    user_clean = user_input.lower().replace(' ', '')
    
    priority_order = [
        'QUALIFICATION',
        'APPLICATION_PERIOD', 
        'APPLICATION_METHOD',
        'CANCEL',
        'CHANGE',
        'PROGRAM_COMPARISON',
        'RECOMMENDATION',
        'CREDIT_INFO',
        'PROGRAM_INFO',
        'COURSE_SEARCH',
        'CONTACT_SEARCH',
        'GREETING',
    ]
    
    for intent in priority_order:
        keywords = INTENT_KEYWORDS.get(intent, [])
        if any(kw in user_clean for kw in keywords):
            return intent
    
    return None


def classify_with_ai(user_input):
    """AI를 사용한 의도 분류"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"질문: {user_input}\n\n의도를 분류하세요.",
            config={
                'system_instruction': INTENT_CLASSIFICATION_PROMPT,
                'temperature': 0,
                'max_output_tokens': 50
            }
        )
        
        intent = response.text.strip().upper()
        
        valid_intents = [
            'QUALIFICATION', 'APPLICATION_PERIOD', 'APPLICATION_METHOD',
            'CANCEL', 'CHANGE', 'PROGRAM_COMPARISON', 'PROGRAM_INFO',
            'CREDIT_INFO', 'COURSE_SEARCH', 'CONTACT_SEARCH',
            'RECOMMENDATION', 'GREETING', 'OUT_OF_SCOPE'
        ]
        
        for valid in valid_intents:
            if valid in intent:
                return valid
        
        # 다전공과 무관한 질문은 OUT_OF_SCOPE
        return 'OUT_OF_SCOPE'
    except:
        return 'OUT_OF_SCOPE'


def classify_intent(user_input, use_ai_fallback=True):
    """
    🔥 Level 2 의도 분류 (Semantic Router 적용!)
    
    🚫 최우선: 욕설/비속어 차단 ← 🆕 추가!
    0단계: 복합 조건 검사 (구체적인 질문 우선)
    1단계: Semantic Router (의미 기반) ← 🆕 핵심!
    2단계: 키워드 매칭 (폴백)
    3단계: AI 분류 (최종 폴백)
    
    Returns: (intent, method, extracted_info)
    """
    user_clean = user_input.lower().replace(' ', '')
    
    # ============================================================
    # 🚫 최우선: 욕설/비속어 차단 (가장 먼저 검사!)
    # ============================================================
    blocked_keywords = INTENT_KEYWORDS.get('BLOCKED', [])
    if any(kw in user_clean for kw in blocked_keywords):
        return 'BLOCKED', 'blocked', {}
    
    # 괄호 표현 정규화
    bracket_pattern = r'([가-힣a-z]+)\(([가-힣a-z]+)\)'
    bracket_match = re.search(bracket_pattern, user_clean)
    if bracket_match:
        inner_term = bracket_match.group(2)
        if inner_term in ['마이크로디그리', '마이크로', 'md', '소단위']:
            user_clean = user_clean.replace(bracket_match.group(0), '마이크로디그리')
    
    # ============================================================
    # 🔥 0단계: 복합 조건 (우선 처리!)
    # ============================================================
    has_year = bool(re.search(r'(20\d{2}|학번|\d{2}학번)', user_clean))
    has_credit_detail = any(kw in user_clean for kw in ['전필', '전선', '이수한', '들은', '수강한'])
    has_recommend = any(kw in user_clean for kw in ['추천', '뭐할까', '어떤게좋', '골라', '뭐가좋', '어떤걸', '뭐해야', '좋을까'])
    has_credit_general = any(kw in user_clean for kw in ['학점', '몇학점'])
    has_major = bool(re.search(r'([가-힣]+(?:학|공학|과학|전공))', user_clean))
    
    # 🆕 교과목/과목 검색 우선 처리 (특정 전공명 + 과목/교과목 키워드)
    has_course_keyword = any(kw in user_clean for kw in ['교과목', '과목', '어떤과목', '무슨과목', '커리큘럼', '수업'])
    if has_course_keyword and has_major:
        return 'COURSE_SEARCH', 'complex', extract_additional_info(user_input, 'COURSE_SEARCH')
    
    # 맞춤형 추천 요청
    if has_recommend:
        if has_year and (has_credit_detail or has_credit_general):
            return 'RECOMMENDATION', 'complex', extract_additional_info(user_input, 'RECOMMENDATION')
        if has_major and (has_credit_detail or has_credit_general):
            return 'RECOMMENDATION', 'complex', extract_additional_info(user_input, 'RECOMMENDATION')
        if has_credit_detail:
            return 'RECOMMENDATION', 'complex', extract_additional_info(user_input, 'RECOMMENDATION')
    
    # 특정 제도 + 특정 질문
    found_programs = extract_programs(user_clean)
    
    if found_programs:
        program = found_programs[0]
        
        if any(kw in user_clean for kw in ['자격', '신청할수있', '가능한지', '조건']):
            return 'QUALIFICATION', 'complex', {'program': program, 'programs': found_programs}
        
        if any(kw in user_clean for kw in ['언제', '기간', '마감']):
            return 'APPLICATION_PERIOD', 'complex', {'program': program, 'programs': found_programs}
        
        if any(kw in user_clean for kw in ['어떻게', '방법', '절차']):
            return 'APPLICATION_METHOD', 'complex', {'program': program, 'programs': found_programs}
        
        if has_credit_general and not has_recommend:
            return 'CREDIT_INFO', 'complex', {'program': program, 'programs': found_programs}
    
    # 제도 비교 특수 처리
    if any(kw in user_clean for kw in INTENT_KEYWORDS.get('PROGRAM_COMPARISON', [])):
        if len(found_programs) >= 2:
            return 'PROGRAM_COMPARISON', 'keyword', {'programs': found_programs}
    
    if '와' in user_clean or '과' in user_clean or '이랑' in user_clean:
        if '과정' not in user_clean:
            if len(found_programs) >= 2:
                return 'PROGRAM_COMPARISON', 'keyword', {'programs': found_programs}
    
    # ============================================================
    # 🔥 1단계: Semantic Router (의미 기반 분류) ← 핵심!
    # ============================================================
    if SEMANTIC_ROUTER is not None:
        semantic_intent, score = classify_with_semantic_router(user_input)
        if semantic_intent:
            extracted_info = extract_additional_info(user_input, semantic_intent)
            return semantic_intent, 'semantic', extracted_info
    
    # ============================================================
    # 🔹 2단계: 키워드 기반 분류 (폴백)
    # ============================================================
    keyword_intent = classify_with_keywords(user_input)
    if keyword_intent:
        extracted_info = extract_additional_info(user_input, keyword_intent)
        return keyword_intent, 'keyword', extracted_info
    
    # 제도 설명 질문
    if found_programs:
        explanation_keywords = ['은?', '는?', '이?', '가?', '뭐', '무엇', '알려', '설명']
        if any(kw in user_clean for kw in explanation_keywords):
            return 'PROGRAM_INFO', 'keyword', {'program': found_programs[0]}
    
    # ============================================================
    # 🔹 3단계: AI 분류 (최종 폴백)
    # ============================================================
    if use_ai_fallback:
        try:
            ai_intent = classify_with_ai(user_input)
            if ai_intent != 'GENERAL':
                extracted_info = extract_additional_info(user_input, ai_intent)
                return ai_intent, 'ai', extracted_info
        except:
            pass
    
    # ============================================================
    # 🚫 최종: 다전공과 무관한 질문 → "모릅니다" 응답
    # ============================================================
    return 'OUT_OF_SCOPE', 'fallback', {}


# ============================================================
# 🎯 핸들러 함수들 (의도별 답변 생성) - v2 스타일 (FAQ 활용)
# ============================================================

def handle_qualification(user_input, extracted_info, data_dict):
    """신청 자격 질문 처리"""
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    response = "## 📋 다전공 제도별 신청 자격 요건\n\n"
    response += "| 제도 | 신청 자격 |\n"
    response += "|------|----------|\n"
    
    for p_name, p_info in programs.items():
        qual = p_info.get('qualification', '-')
        response += f"| **{p_name}** | {qual} |\n"
    
    response += "\n---\n"
    response += "💡 **참고**: 신청 자격은 학칙 개정에 따라 변경될 수 있습니다.\n\n"
    response += CONTACT_MESSAGE
    
    return response, "QUALIFICATION"


def handle_application_period(user_input, extracted_info, data_dict):
    """신청 기간 질문 처리"""
    faq_data = data_dict.get('faq', FAQ_DATA)
    
    # FAQ에서 관련 정보 검색
    period_answer = None
    for faq in faq_data:
        q = faq.get('질문', '').lower().replace(' ', '')
        if '신청' in q and ('기간' in q or '언제' in q):
            period_answer = faq.get('답변', '')
            break
    
    response = f"## {APP_PERIOD_TITLE}\n\n"
    response += f"{APP_PERIOD_INTRO}\n\n"
    response += "### 📌 신청 시기\n\n"
    response += "| 이수 희망 학기 | 신청 시기 |\n"
    response += "|--------------|----------|\n"
    response += f"| **1학기** 이수 희망 | {APP_PERIOD_1ST} |\n"
    response += f"| **2학기** 이수 희망 | {APP_PERIOD_2ND} |\n\n"
    
    response += "### ⏰ 신청 가능 시점\n"
    response += f"- {APP_PERIOD.get('start_info', '**입학 후 첫 학기부터** 신청 가능합니다.')}\n"
    response += f"- {APP_PERIOD.get('restriction', '졸업 예정 학기에는 신청이 제한될 수 있습니다.')}\n\n"
    
    if period_answer:
        response += f"### 📋 참고 정보\n{period_answer}\n\n"
    
    response += "---\n"
    response += f"⚠️ 정확한 일정은 학교 홈페이지 **[학사공지]({ACADEMIC_NOTICE_URL})**를 확인하세요.\n\n"
    response += CONTACT_MESSAGE
    
    return response, "APPLICATION_PERIOD"


def handle_application_method(user_input, extracted_info, data_dict):
    """신청 방법/절차 질문 처리"""
    faq_data = data_dict.get('faq', FAQ_DATA)
    
    # FAQ에서 관련 정보 검색
    method_answers = []
    for faq in faq_data:
        q = faq.get('질문', '').lower().replace(' ', '')
        if ('신청' in q or '지원' in q) and ('방법' in q or '절차' in q or '어떻게' in q):
            method_answers.append({
                'question': faq.get('질문', ''),
                'answer': faq.get('답변', '')
            })
    
    response = "## 📝 다전공 신청 방법 안내\n\n"
    
    if method_answers:
        for item in method_answers[:3]:
            response += f"**Q. {item['question']}**\n\n"
            response += f"A. {item['answer']}\n\n"
            response += "---\n\n"
    else:
        response += "**일반적인 신청 절차:**\n\n"
        response += "1️⃣ **신청 시기 확인**: 학사 공지사항에서 신청 기간 확인\n\n"
        response += "2️⃣ **자격 요건 확인**: 본인의 학년, 평점 등 자격 충족 여부 확인\n\n"
        response += "3️⃣ **온라인 신청**: 학사공지에 안내된 방법으로 신청서 작성\n\n"
        response += "4️⃣ **승인 대기**: 해당 학과에서 승인 절차 진행\n\n"
        response += "---\n\n"
    
    response += "⚠️ 자세한 내용은 학교 홈페이지 **[학사공지](https://www.hknu.ac.kr/kor/562/subview.do)**를 참고하거나\n\n"
    response += CONTACT_MESSAGE
    
    return response, "APPLICATION_METHOD"


def handle_cancel(user_input, extracted_info, data_dict):
    """포기/취소 질문 처리"""
    faq_data = data_dict.get('faq', FAQ_DATA)
    
    # FAQ에서 포기/취소 관련 정보 검색
    cancel_answers = []
    for faq in faq_data:
        q = faq.get('질문', '').lower()
        if '포기' in q or '취소' in q or '철회' in q:
            cancel_answers.append({
                'question': faq.get('질문', ''),
                'answer': faq.get('답변', '')
            })
    
    response = "## ❌ 다전공 포기/취소 안내\n\n"
    
    if cancel_answers:
        for item in cancel_answers[:3]:
            response += f"**Q. {item['question']}**\n\n"
            response += f"A. {item['answer']}\n\n"
            response += "---\n\n"
    else:
        response += "**다전공 포기 안내:**\n\n"
        response += "- **포기 시기**: 매 학기 수강신청 기간 중 가능\n"
        response += "- **포기 방법**: 학사공지 확인 후 신청\n"
        response += "- **유의사항**: 이수한 학점은 자유선택 학점으로 인정\n\n"
        response += "---\n\n"
    
    response += "⚠️ 자세한 내용은 학교 홈페이지 **[학사공지](https://www.hknu.ac.kr/kor/562/subview.do)**를 참고하거나\n\n"
    response += CONTACT_MESSAGE
    
    return response, "CANCEL"


def handle_change(user_input, extracted_info, data_dict):
    """변경 질문 처리"""
    faq_data = data_dict.get('faq', FAQ_DATA)
    
    # FAQ에서 변경 관련 정보 검색
    change_answers = []
    for faq in faq_data:
        q = faq.get('질문', '').lower()
        if '변경' in q or '수정' in q or '바꾸' in q or '전환' in q:
            change_answers.append({
                'question': faq.get('질문', ''),
                'answer': faq.get('답변', '')
            })
    
    response = "## 🔄 다전공 변경 안내\n\n"
    
    if change_answers:
        for item in change_answers[:3]:
            response += f"**Q. {item['question']}**\n\n"
            response += f"A. {item['answer']}\n\n"
            response += "---\n\n"
    else:
        response += "**다전공 변경 안내:**\n\n"
        response += "- 다전공 **종류 변경** (예: 복수전공 → 부전공): 기존 포기 후 재신청\n"
        response += "- 다전공 **전공 변경** (예: A전공 → B전공): 기존 포기 후 재신청\n\n"
        response += "※ 동일 학기에 포기와 신청을 동시에 처리할 수 있습니다.\n\n"
        response += "---\n\n"
    
    response += "⚠️ 자세한 내용은 학교 홈페이지 **[학사공지](https://www.hknu.ac.kr/kor/562/subview.do)**를 참고하거나\n\n"
    response += CONTACT_MESSAGE
    
    return response, "CHANGE"


def handle_program_comparison(user_input, extracted_info, data_dict):
    """제도 비교 질문 처리"""
    programs_to_compare = extracted_info.get('programs', [])
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    if len(programs_to_compare) < 2:
        programs_to_compare = list(programs.keys())[:4]
    
    comparison_data = []
    for program_name in programs_to_compare:
        if program_name in programs:
            comparison_data.append({
                'name': program_name,
                **programs[program_name]
            })
        elif program_name == '마이크로디그리' and '소단위전공과정' in programs:
            comparison_data.append({
                'name': '소단위전공과정(마이크로디그리)',
                **programs['소단위전공과정']
            })
    
    if len(comparison_data) < 2:
        response = "## 📊 다전공 제도 비교\n\n"
        response += "| 구분 | 복수전공 | 부전공 | 융합전공 | 마이크로디그리 |\n"
        response += "|------|----------|--------|----------|----------------|\n"
        response += "| **이수학점** | 36학점 이상 | 21학점 이상 | 36학점 이상 | 12학점 |\n"
        response += "| **학위표기** | 2개 학위 | 부전공 표기 | 융합전공명 | 이수증 |\n"
        response += "| **본전공 감축** | 있음 | 있음 | 있음 | 없음 |\n"
        response += "| **난이도** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |\n\n"
        response += CONTACT_MESSAGE
        return response, "PROGRAM_COMPARISON"
    
    response = f"## 📊 {' vs '.join([d['name'] for d in comparison_data])} 비교\n\n"
    response += "| 구분 | " + " | ".join([d['name'] for d in comparison_data]) + " |\n"
    response += "|------" + "|------" * len(comparison_data) + "|\n"
    
    response += "| **이수학점** | " + " | ".join([d.get('credits_multi', '-') for d in comparison_data]) + " |\n"
    response += "| **본전공** | " + " | ".join([d.get('credits_primary', '-') for d in comparison_data]) + " |\n"
    
    quals = []
    for d in comparison_data:
        q = d.get('qualification', '-')
        quals.append(q[:15] + '...' if len(q) > 15 else q)
    response += "| **신청자격** | " + " | ".join(quals) + " |\n"
    
    response += "| **학위표기** | " + " | ".join([str(d.get('degree', '-')) for d in comparison_data]) + " |\n"
    response += "| **난이도** | " + " | ".join([str(d.get('difficulty', '-')) for d in comparison_data]) + " |\n"
    
    response += "\n---\n"
    response += CONTACT_MESSAGE
    
    return response, "PROGRAM_COMPARISON"


def handle_credit_info(user_input, extracted_info, data_dict):
    """학점 정보 질문 처리"""
    primary_req = data_dict.get('primary_req', PRIMARY_REQ)
    grad_req = data_dict.get('grad_req', GRADUATION_REQ)
    
    response = "## 📖 다전공 제도별 이수 학점\n\n"
    response += "⚠️ **전공필수/전공선택 학점은 본전공과 학번에 따라 다를 수 있습니다.**\n\n"
    
    # 제도 유형 목록
    program_types = ["복수전공", "부전공", "융합전공", "융합부전공", "연계전공"]
    
    # 2025학번 경영학전공 예시
    response += "### 📌 예시: 2025학번 경영학전공 기준\n\n"
    
    response += "#### 📚 본전공 이수 학점 (다전공 신청 시 변경)\n\n"
    response += "| 제도 | 전공필수 | 전공선택 | 계 |\n"
    response += "|------|----------|----------|----|\n"
    
    if not primary_req.empty:
        for p_type in program_types:
            # 경영학전공 + 2025학번 기준 필터링
            filtered = primary_req[
                (primary_req['제도유형'].str.contains(p_type, na=False)) &
                (primary_req['전공명'].str.contains('경영학', na=False)) &
                (primary_req['기준학번'] == 2025)
            ]
            if filtered.empty:
                # 2025학번이 없으면 가장 최근 학번
                filtered = primary_req[
                    (primary_req['제도유형'].str.contains(p_type, na=False)) &
                    (primary_req['전공명'].str.contains('경영학', na=False))
                ]
            if not filtered.empty:
                row = filtered.sort_values('기준학번', ascending=False).iloc[0]
                req = row.get('본전공_전공필수', '-')
                elec = row.get('본전공_전공선택', '-')
                total = row.get('본전공_계', '-')
                response += f"| **{p_type}** | {req} | {elec} | {total} |\n"
    else:
        response += "| - | 데이터 없음 | - | - |\n"
    
    response += "\n#### 🎓 다전공 이수 학점\n\n"
    response += "| 제도 | 전공필수 | 전공선택 | 계 |\n"
    response += "|------|----------|----------|----|\n"
    
    if not grad_req.empty:
        for p_type in program_types:
            filtered = grad_req[grad_req['제도유형'].str.contains(p_type, na=False)]
            if not filtered.empty:
                row = filtered.sort_values('기준학번', ascending=False).iloc[0]
                req = row.get('다전공_전공필수', '-')
                elec = row.get('다전공_전공선택', '-')
                total = row.get('다전공_계', '-')
                response += f"| **{p_type}** | {req} | {elec} | {total} |\n"
    else:
        response += "| - | 데이터 없음 | - | - |\n"
    
    response += "\n---\n"
    response += "💡 **참고**: 다전공 신청 시 본전공 이수 학점이 줄어들 수 있습니다. 본인의 전공과 학번에 맞는 정확한 학점은 왼쪽 메뉴의 '다전공 제도 안내'에서 확인하세요.\n\n"
    response += "⚠️ 기타 내용은 학교 홈페이지 **[학사공지](https://www.hknu.ac.kr/kor/562/subview.do)**를 참고하거나\n\n"
    response += CONTACT_MESSAGE
    
    return response, "CREDIT_INFO"


def handle_program_info(user_input, extracted_info, data_dict):
    """제도 설명 질문 처리"""
    program_name = extracted_info.get('program', '')
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    program_mapping = {
        '복수전공': '복수전공',
        '부전공': '부전공',
        '융합전공': '융합전공',
        '융합부전공': '융합부전공',
        '연계전공': '연계전공',
        '마이크로디그리': '소단위전공과정',
    }
    
    actual_name = program_mapping.get(program_name, program_name)
    
    if actual_name not in programs:
        for key in programs.keys():
            if program_name in key or key in program_name:
                actual_name = key
                break
    
    if actual_name not in programs:
        return f"'{program_name}' 제도 정보를 찾을 수 없습니다.\n📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    info = programs[actual_name]
    display_name = actual_name
    if actual_name == '소단위전공과정':
        display_name = '소단위전공과정(마이크로디그리)'
    
    features_text = '\n'.join([f"- {f.strip()}" for f in info.get('features', [])]) if info.get('features') else '없음'
    
    response = f"## 🎓 {display_name}\n\n"
    response += f"### 📖 개요\n{info.get('description', '-')}\n\n"
    
    response += "### 📚 이수학점\n"
    response += "| 구분 | 학점 |\n"
    response += "|------|------|\n"
    response += f"| 교양 | {info.get('credits_general', '-')} |\n"
    response += f"| 원전공(본전공) | {info.get('credits_primary', '-')} |\n"
    response += f"| 다전공 | {info.get('credits_multi', '-')} |\n\n"
    
    response += f"### ✅ 신청자격\n{info.get('qualification', '-')}\n\n"
    response += f"### 📜 학위표기\n{info.get('degree', '-')}\n\n"
    response += f"### ⭐ 난이도\n{info.get('difficulty', '-')}\n\n"
    response += f"### ✨ 특징\n{features_text}\n\n"
    
    if info.get('notes'):
        response += f"### 💡 유의사항\n{info['notes']}\n\n"
    
    response += "---\n"
    response += CONTACT_MESSAGE
    
    return response, "PROGRAM_INFO"


def handle_course_search(user_input, extracted_info, data_dict):
    """과목 조회 질문 처리"""
    major = extracted_info.get('major')
    year = extracted_info.get('year')
    
    courses_data = data_dict.get('courses', COURSES_DATA)
    
    # 전공명이 없으면 입력에서 직접 찾기
    if not major and not courses_data.empty:
        user_clean = user_input.replace(' ', '')
        for m in courses_data['전공명'].unique():
            m_clean = str(m).replace(' ', '')
            if m_clean in user_clean or user_clean in m_clean:
                major = m
                break
            # 부분 매칭도 시도 (예: "AI반도체" -> "AI반도체융합전공")
            if len(m_clean) > 3:
                keyword = m_clean.replace('전공', '').replace('융합', '')[:4]
                if keyword in user_clean:
                    major = m
                    break
    
    if not major:
        return """## 📚 과목 조회

어떤 전공의 과목을 찾으시나요?

💡 **예시 질문:**
- "AI반도체융합전공 어떤 과목 들어?"
- "빅데이터융합전공 교과목 알려줘"
- "소프트웨어융합전공 과목 보여줘"

📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.""", "COURSE_SEARCH"
    
    if courses_data.empty:
        return f"'{major}' 과목 정보를 찾을 수 없습니다.\n📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    # 전공명으로 필터링 (정확한 매칭 우선, 없으면 부분 매칭)
    major_courses = courses_data[courses_data['전공명'] == major]
    
    if major_courses.empty:
        # 부분 매칭 시도
        major_keyword = major.replace('전공', '').replace('융합', '')
        major_courses = courses_data[
            courses_data['전공명'].str.contains(major_keyword, case=False, na=False)
        ]
    
    if major_courses.empty:
        return f"'{major}' 과목 정보를 찾을 수 없습니다.\n📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    # 제도유형 정보 표시
    program_types = major_courses['제도유형'].unique().tolist()
    
    if year:
        major_courses = major_courses[
            (major_courses['학년'] == year) |
            (major_courses['학년'].astype(str) == str(year))
        ]
    
    if major_courses.empty:
        return f"'{major}' {year}학년 과목 정보가 없습니다.\n📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    # 실제 전공명 가져오기
    actual_major = major_courses['전공명'].iloc[0]
    
    response = f"## 📚 {actual_major} 교과목 안내\n\n"
    response += f"📋 **제도유형**: {', '.join([str(pt) for pt in program_types if pd.notna(pt)])}\n\n"
    
    years_in_data = sorted([int(y) for y in major_courses['학년'].dropna().unique()])
    
    for y in years_in_data:
        year_data = major_courses[major_courses['학년'] == y]
        response += f"### {y}학년\n\n"
        
        for _, row in year_data.iterrows():
            sem = row.get('학기', '-')
            course_type = row.get('이수구분', '-')
            course_name = row.get('과목명', '-')
            credit = row.get('학점', '-')
            
            if '필수' in str(course_type):
                badge = "🔴"
            elif '선택' in str(course_type):
                badge = "🟢"
            else:
                badge = "🔵"
            
            try:
                credit_str = f"{int(credit)}학점"
            except:
                credit_str = f"{credit}학점" if pd.notna(credit) else ""
            
            try:
                sem_str = f"{int(sem)}학기"
            except:
                sem_str = f"{sem}" if pd.notna(sem) else ""
            
            response += f"{badge} [{course_type}] {course_name} ({credit_str}) - {sem_str}\n"
        
        response += "\n"
    
    response += "---\n"
    response += CONTACT_MESSAGE
    
    return response, "COURSE_SEARCH"


def handle_contact_search(user_input, extracted_info, data_dict):
    """연락처 조회 질문 처리"""
    major = extracted_info.get('major')
    majors_info = data_dict.get('majors', MAJORS_INFO)
    
    if majors_info.empty:
        return "전공 정보를 불러올 수 없습니다.\n📞 문의: 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    if not major:
        user_clean = user_input.replace(' ', '')
        for _, row in majors_info.iterrows():
            m_name = str(row['전공명'])
            if m_name.replace(' ', '') in user_clean or user_clean in m_name.replace(' ', ''):
                major = m_name
                break
    
    if not major:
        return """## 📞 연락처 조회

어떤 전공의 연락처를 찾으시나요?

💡 **예시 질문:**
- "경영학전공 연락처 알려줘"
- "소프트웨어융합전공 사무실 위치"

📞 문의: 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.""", "CONTACT_SEARCH"
    
    result = majors_info[majors_info['전공명'].str.contains(major.replace('전공', ''), case=False, na=False)]
    
    if result.empty:
        return f"'{major}' 연락처를 찾을 수 없습니다.\n📞 문의: 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"
    
    row = result.iloc[0]
    
    response = f"## 📞 {row['전공명']} 연락처\n\n"
    response += "| 항목 | 정보 |\n"
    response += "|------|------|\n"
    response += f"| **전공명** | {row['전공명']} |\n"
    response += f"| **연락처** | {row.get('연락처', '-')} |\n"
    response += f"| **위치** | {row.get('위치', '-')} |\n"
    
    homepage = row.get('홈페이지', '-')
    if pd.notna(homepage) and homepage != '-':
        response += f"| **홈페이지** | [{homepage}]({homepage}) |\n"
    
    return response, "CONTACT_SEARCH"


def handle_recommendation(user_input, extracted_info, data_dict):
    """추천 질문 처리"""
    user_info = extract_user_info_for_recommendation(user_input, data_dict)
    
    if user_info.get('has_all_info'):
        result = calculate_multi_major_recommendation(
            user_info['admission_year'],
            user_info['primary_major'],
            user_info['completed_required'],
            user_info['completed_elective'],
            data_dict
        )
        return result, "RECOMMENDATION"
    else:
        missing = user_info.get('missing', [])
        
        response = "## 🎯 맞춤형 다전공 추천\n\n"
        response += "정확한 추천을 위해 아래 정보가 필요합니다:\n\n"
        
        if 'admission_year' in missing:
            response += "- **기준학번** (예: 2022학번)\n"
        if 'primary_major' in missing:
            response += "- **현재 본전공** (예: 경영학전공)\n"
        if 'completed_required' in missing:
            response += "- **이수한 전공필수 학점**\n"
        if 'completed_elective' in missing:
            response += "- **이수한 전공선택 학점**\n"
        
        response += "\n💡 **예시 질문:**\n"
        response += '"저는 2022학번 경영학전공이고, 전공필수 3학점, 전공선택 9학점 들었어요. 다전공 추천해주세요!"\n\n'
        response += CONTACT_MESSAGE
        
        return response, "RECOMMENDATION"


def calculate_multi_major_recommendation(admission_year, primary_major, completed_required, completed_elective, data_dict):
    """학생의 이수 현황을 바탕으로 다전공 추천"""
    
    result = "## 🎓 맞춤형 다전공 추천 결과\n\n"
    result += f"**📋 입력 정보**\n"
    result += f"- 기준학번: {admission_year}학번\n"
    result += f"- 본전공: {primary_major}\n"
    result += f"- 이수 현황: 전필 {completed_required}학점, 전선 {completed_elective}학점 (총 {completed_required + completed_elective}학점)\n\n"
    
    primary_req = data_dict.get('primary_req', PRIMARY_REQ)
    grad_req = data_dict.get('grad_req', GRADUATION_REQ)
    
    if primary_req.empty:
        return result + "⚠️ 본전공 이수요건 데이터가 없어 추천이 불가능합니다."
    
    # 본전공 데이터 필터링
    primary_data = primary_req[primary_req['전공명'] == primary_major].copy()
    
    if primary_data.empty:
        return result + f"⚠️ '{primary_major}' 전공의 이수요건 데이터를 찾을 수 없습니다."
    
    primary_data['기준학번'] = pd.to_numeric(primary_data['기준학번'], errors='coerce')
    applicable_primary = primary_data[primary_data['기준학번'] <= admission_year]
    
    if applicable_primary.empty:
        return result + f"⚠️ {admission_year}학번에 해당하는 본전공 이수요건을 찾을 수 없습니다."
    
    # 제도별 분석
    programs_to_analyze = ["복수전공", "부전공", "융합전공", "융합부전공", "연계전공"]
    recommendations = []
    
    result += "### 📊 제도별 학점 분석\n\n"
    result += "| 제도 | 본전공 변경 | 남은 본전공 | 다전공 이수 | 총 추가 학점 | 평가 |\n"
    result += "|------|------------|-----------|-----------|------------|------|\n"
    
    for program in programs_to_analyze:
        program_primary = applicable_primary[applicable_primary['제도유형'].str.contains(program, na=False)]
        
        if program_primary.empty:
            continue
        
        program_primary = program_primary.sort_values('기준학번', ascending=False)
        primary_row = program_primary.iloc[0]
        
        new_primary_required = int(primary_row.get('본전공_전공필수', 0))
        new_primary_elective = int(primary_row.get('본전공_전공선택', 0))
        new_primary_total = int(primary_row.get('본전공_계', 0))
        
        remaining_primary_required = max(0, new_primary_required - completed_required)
        remaining_primary_elective = max(0, new_primary_elective - completed_elective)
        remaining_primary_total = remaining_primary_required + remaining_primary_elective
        
        # 다전공 기본 학점
        multi_credits = {
            "복수전공": 36,
            "부전공": 21,
            "융합전공": 36,
            "융합부전공": 21,
            "연계전공": 36
        }
        multi_total = multi_credits.get(program, 36)
        
        total_remaining = remaining_primary_total + multi_total
        
        if total_remaining <= 40:
            rating = "🟢 매우 유리"
        elif total_remaining <= 55:
            rating = "🟡 보통"
        else:
            rating = "🔴 부담 큼"
        
        recommendations.append({
            'program': program,
            'remaining_primary_total': remaining_primary_total,
            'multi_total': multi_total,
            'total_remaining': total_remaining,
            'rating': rating
        })
        
        result += f"| {program} | {new_primary_total}학점 | {remaining_primary_total}학점 | {multi_total}학점 | **{total_remaining}학점** | {rating} |\n"
    
    result += "\n"
    
    if recommendations:
        recommendations.sort(key=lambda x: x['total_remaining'])
        
        result += "### 🌟 추천 순위\n\n"
        for idx, rec in enumerate(recommendations[:3], 1):
            result += f"**{idx}순위: {rec['program']}** - 총 {rec['total_remaining']}학점 {rec['rating']}\n"
        
        result += "\n"
    
    result += "### 🎯 마이크로디그리 (소단위전공) - 추가 추천\n"
    result += "- **특징**: 본전공 학점 감면 없음\n"
    result += "- **추가 학점**: 12~18학점\n"
    result += "- **장점**: 다른 다전공과 병행 가능\n\n"
    
    result += "---\n"
    result += "⚠️ 자세한 내용은 학교 홈페이지 **[학사공지](https://www.hknu.ac.kr/kor/562/subview.do)**를 참고하거나\n\n"
    result += CONTACT_MESSAGE
    
    return result


def extract_user_info_for_recommendation(user_input, data_dict):
    """추천을 위한 사용자 정보 추출"""
    user_info = {'missing': []}
    
    majors_list = []
    if 'primary_req' in data_dict and not data_dict['primary_req'].empty:
        majors_list = data_dict['primary_req']['전공명'].unique().tolist()
    
    year_match = re.search(r'(20\d{2})[학년번]|(\d{2})[학년번]', user_input)
    if year_match:
        year = year_match.group(1) if year_match.group(1) else f"20{year_match.group(2)}"
        user_info['admission_year'] = int(year)
    else:
        user_info['missing'].append('admission_year')
    
    for major in majors_list:
        if major in user_input:
            user_info['primary_major'] = major
            break
    
    if 'primary_major' not in user_info:
        major_pattern = r'([가-힣]+(?:학|공학|과학))전공'
        major_matches = re.findall(major_pattern, user_input)
        if major_matches:
            user_info['primary_major'] = major_matches[0] + "전공"
        else:
            user_info['missing'].append('primary_major')
    
    required_patterns = [
        r'전[공]?필[수]?\s*(\d+)\s*학점',
        r'필수\s*(\d+)\s*학점',
        r'전필\s*(\d+)',
        r'전공필수\s*(\d+)',
    ]
    for pattern in required_patterns:
        match = re.search(pattern, user_input)
        if match:
            user_info['completed_required'] = int(match.group(1))
            break
    if 'completed_required' not in user_info:
        user_info['missing'].append('completed_required')
    
    elective_patterns = [
        r'전[공]?선[택]?\s*(\d+)\s*학점',
        r'선택\s*(\d+)\s*학점',
        r'전선\s*(\d+)',
        r'전공선택\s*(\d+)',
    ]
    for pattern in elective_patterns:
        match = re.search(pattern, user_input)
        if match:
            user_info['completed_elective'] = int(match.group(1))
            break
    if 'completed_elective' not in user_info:
        user_info['missing'].append('completed_elective')
    
    user_info['has_all_info'] = len(user_info['missing']) == 0
    
    return user_info


def handle_greeting(user_input, extracted_info, data_dict):
    """인사 처리"""
    response = """## 👋 안녕하세요!

**한경국립대학교 다전공(유연학사제도) 안내 AI챗봇**입니다.

---

### 🎯 무엇을 도와드릴까요?

| 질문 유형 | 예시 |
|----------|------|
| 📝 신청 | "신청 자격이 뭐야?" / "언제 신청해?" |
| 📊 비교 | "복수전공이랑 부전공 차이점" |
| 📖 학점 | "부전공 몇 학점 들어야 해?" |
| 🎯 추천 | "나한테 맞는 다전공 추천해줘" |
| 📞 연락처 | "경영학전공 사무실 번호" |

---

💡 **Tip**: 위의 **'💡 어떤 질문을 해야 할지 모르겠나요?'**를 클릭하면 예시 질문을 바로 선택할 수 있어요!

무엇이든 물어보세요! 😊"""
    
    return response, "GREETING"


def handle_blocked(user_input, extracted_info, data_dict):
    """욕설/부적절한 질문 차단"""
    response = """## ⚠️ 잠깐만요!

부적절한 표현이 감지되었어요.

저는 **한경국립대학교 학생들을 돕기 위한 AI챗봇**이에요.
다전공 관련 질문을 해주시면 친절하게 답변드릴게요! 😊

---

💡 **이런 질문은 어떠세요?**
- "복수전공 신청 자격이 뭐야?"
- "부전공이랑 복수전공 차이점 알려줘"
- "경영학전공 연락처 알려줘"

"""
    return response, "BLOCKED"


def handle_out_of_scope(user_input, extracted_info, data_dict):
    """범위 외 질문 처리 - 다전공과 무관한 질문"""
    response = """## 🚫 모릅니다

저는 **한경국립대학교 다전공(유연학사제도) 전용 AI챗봇**이에요.
해당 질문은 제가 답변드리기 어려워요.

---

### 💬 이런 질문은 답변할 수 있어요!

| 카테고리 | 질문 예시 |
|---------|----------|
| 📝 **신청 관련** | 신청 자격이 뭐야? / 신청 기간 언제야? / 어떻게 신청해? |
| 🔄 **변경/포기** | 다전공 포기하려면? / 복수전공에서 부전공으로 바꿀 수 있어? |
| 📊 **제도 비교** | 복수전공이랑 부전공 차이가 뭐야? / 융합전공이 뭐야? |
| 📖 **학점 정보** | 복수전공 몇 학점이야? / 본전공 학점 변해? |
| 🎯 **맞춤 추천** | 2022학번 경영학전공인데 다전공 추천해줘 |
| 📞 **연락처** | 경영학전공 사무실 전화번호 / 컴퓨터공학전공 연락처 |
| 📚 **교과목** | 소프트웨어융합전공 어떤 과목 들어? |

---

### 🎈 빠른 시작

👆 **위의 '💡 어떤 질문을 해야 할지 모르겠나요?'**를 클릭해서 예시 질문을 선택해보세요!

**사이드바 메뉴**에서도 다음을 이용할 수 있어요:
- 📊 **'다전공 제도 안내'** → 제도별 상세 정보 확인
- ❓ **'FAQ'** → 자주 묻는 질문 검색

"""
    return response, "OUT_OF_SCOPE"


def get_ai_context(user_input, data_dict):
    """AI 컨텍스트 생성 (RAG)"""
    context = ""
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    for p_name, p_info in programs.items():
        context += f"\n[{p_name}]\n"
        context += f"- 설명: {p_info.get('description', '-')}\n"
        context += f"- 이수학점: {p_info.get('credits_multi', '-')}\n"
        context += f"- 신청자격: {p_info.get('qualification', '-')}\n"
    
    return context


def get_faq_context(user_input, data_dict):
    """FAQ 컨텍스트 생성"""
    faq_data = data_dict.get('faq', FAQ_DATA)
    
    if not faq_data:
        return ""
    
    context = "\n[관련 FAQ]\n"
    count = 0
    
    for faq in faq_data:
        q = faq.get('질문', '')
        a = faq.get('답변', '')
        
        q_clean = q.replace(' ', '').lower()
        if any(kw in q_clean for kw in ['신청', '자격', '학점', '기간', '방법', '포기', '변경']):
            context += f"Q: {q}\nA: {a}\n\n"
            count += 1
            if count >= 3:
                break
    
    return context if count > 0 else ""


def handle_general(user_input, extracted_info, data_dict):
    """일반 질문 처리 - AI에게 위임"""
    context = get_ai_context(user_input, data_dict)
    faq_context = get_faq_context(user_input, data_dict)
    
    prompt = f"""당신은 한경국립대학교 다전공 안내 AI입니다.

학생 질문: {user_input}

[참고 데이터]
{context[:4000] if context else "없음"}

{faq_context if faq_context else ""}

💡 **규칙:**
1. 반드시 '~습니다', '~합니다'체 사용
2. 데이터에 있는 정보만 사용
3. 모르는 내용은 "전공 사무실 또는 학사지원팀(031-670-5035)으로 문의해주세요"
4. 간결하게 200자 이내로 답변
"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'system_instruction': '한경국립대학교 다전공 안내 AI입니다. 친절하고 정확하게 답변하세요.',
                'temperature': 0.3,
            }
        )
        if response and response.text:
            return response.text, "GENERAL"
    except Exception as e:
        pass
    
    return "죄송합니다. 답변을 생성하지 못했습니다.\n📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.", "ERROR"


# 핸들러 매핑
INTENT_HANDLERS = {
    'QUALIFICATION': handle_qualification,
    'APPLICATION_PERIOD': handle_application_period,
    'APPLICATION_METHOD': handle_application_method,
    'CANCEL': handle_cancel,
    'CHANGE': handle_change,
    'PROGRAM_COMPARISON': handle_program_comparison,
    'CREDIT_INFO': handle_credit_info,
    'PROGRAM_INFO': handle_program_info,
    'COURSE_SEARCH': handle_course_search,
    'CONTACT_SEARCH': handle_contact_search,
    'RECOMMENDATION': handle_recommendation,
    'GREETING': handle_greeting,
    'BLOCKED': handle_blocked,
    'OUT_OF_SCOPE': handle_out_of_scope,
    'GENERAL': handle_general,
}



def generate_ai_response(user_input, chat_history, data_dict):
    """통합 응답 생성"""
    intent, method, extracted_info = classify_intent(user_input)
    
    # 디버그 정보 (개발용)
    # st.caption(f"🔍 의도: {intent} | 분류방법: {method}")
    
    handler = INTENT_HANDLERS.get(intent, handle_general)
    response, response_type = handler(user_input, extracted_info, data_dict)
    
    return response, response_type


# ============================================================
# 📊 이수체계도 및 과목 표시 함수
# ============================================================

def display_curriculum_image(major, program_type):
    """이수체계도/과정 안내 이미지 표시
    - 융합전공: 이수체계도 이미지
    - 소단위전공과정(마이크로디그리): 과정 안내 이미지
    """
    # 융합전공이나 소단위전공과정(마이크로디그리)만 이미지 표시
    is_fusion = program_type == "융합전공"
    is_micro = "소단위" in program_type or "마이크로" in program_type
    
    if not is_fusion and not is_micro:
        return
    
    if CURRICULUM_MAPPING.empty:
        return
    
    # 제도유형 매칭
    def match_program_type_for_image(type_value):
        type_str = str(type_value).strip().lower()
        
        if is_fusion:
            return "융합전공" in type_str and "융합부전공" not in type_str
        
        if is_micro:
            return any(kw in type_str for kw in ['소단위', '마이크로', 'md'])
        
        return False
    
    # 전공명 정제 (학부명 제거)
    clean_major = major
    if ' ' in major:
        parts = major.split(' ')
        if len(parts) >= 2 and '학부' in parts[0]:
            clean_major = ' '.join(parts[1:])
    
    # 1. 정확한 매칭 시도
    filtered = CURRICULUM_MAPPING[
        (CURRICULUM_MAPPING['전공명'] == clean_major) & 
        (CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image))
    ]
    
    # 2. 원본 전공명으로 시도
    if filtered.empty and clean_major != major:
        filtered = CURRICULUM_MAPPING[
            (CURRICULUM_MAPPING['전공명'] == major) & 
            (CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image))
        ]
    
    # 3. 부분 매칭 시도 (전공명에서 핵심 키워드 추출)
    if filtered.empty:
        # "스마트팜전문가과정" -> "스마트팜"
        keywords = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '')
        if len(keywords) >= 2:
            filtered = CURRICULUM_MAPPING[
                (CURRICULUM_MAPPING['전공명'].str.contains(keywords[:4], na=False)) & 
                (CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image))
            ]
    
    # 4. 제도유형만으로 전공명 찾기 (curriculum_mapping에서)
    if filtered.empty:
        type_matched = CURRICULUM_MAPPING[CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image)]
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            # 선택한 전공명과 curriculum_mapping의 전공명이 서로 포함 관계인지 확인
            if clean_major in cm_major or cm_major in clean_major:
                filtered = type_matched[type_matched['전공명'] == cm_major]
                break
            # 키워드 비교
            cm_keyword = cm_major.replace('전공', '').replace('과정', '')[:4]
            clean_keyword = clean_major.replace('전공', '').replace('과정', '')[:4]
            if cm_keyword == clean_keyword:
                filtered = type_matched[type_matched['전공명'] == cm_major]
                break
    
    if not filtered.empty:
        filename = filtered.iloc[0]['파일명']
        if pd.notna(filename) and str(filename).strip():
            image_path = f"{CURRICULUM_IMAGES_PATH}/{filename}"
            if os.path.exists(image_path):
                if is_fusion:
                    caption = f"{clean_major} 이수체계도"
                else:
                    caption = f"{clean_major} 과정 안내"
                st.image(image_path, caption=caption)
            else:
                st.caption(f"📷 이미지 파일 준비 중: {filename}")


def display_courses(major, program_type):
    """과목 정보 표시 - 학년별/학기별/이수구분별 정리 + 연락처"""
    if COURSES_DATA.empty:
        st.info("교과목 데이터가 없습니다.")
        return False
    
    # 제도유형 매칭 함수
    is_micro = "소단위" in program_type or "마이크로" in program_type
    
    def match_program_type_for_courses(type_value):
        type_str = str(type_value).strip().lower()
        
        if is_micro:
            return any(kw in type_str for kw in ['소단위', '마이크로', 'md'])
        
        if program_type == "부전공":
            return "부전공" in type_str and "융합부전공" not in type_str
        
        if program_type == "융합전공":
            return "융합전공" in type_str and "융합부전공" not in type_str
        
        return program_type in type_str
    
    # 전공명 정제 (학부명 제거)
    clean_major = major
    if ' ' in major:
        parts = major.split(' ')
        if len(parts) >= 2 and '학부' in parts[0]:
            clean_major = ' '.join(parts[1:])
    
    # 1. 정확한 매칭
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == clean_major) & 
        (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
    ]
    
    # 2. 원본 전공명으로 시도
    if courses.empty and clean_major != major:
        courses = COURSES_DATA[
            (COURSES_DATA['전공명'] == major) & 
            (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
        ]
    
    # 3. 부분 매칭 (전공명 키워드)
    if courses.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '')
        if len(keyword) >= 2:
            courses = COURSES_DATA[
                (COURSES_DATA['전공명'].str.contains(keyword[:4], na=False)) & 
                (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
            ]
    
    # 4. 제도유형으로 먼저 필터링 후 전공명 찾기
    if courses.empty:
        type_matched = COURSES_DATA[COURSES_DATA['제도유형'].apply(match_program_type_for_courses)]
        for course_major in type_matched['전공명'].unique():
            cm_str = str(course_major)
            if clean_major in cm_str or cm_str in clean_major:
                courses = type_matched[type_matched['전공명'] == course_major]
                clean_major = cm_str  # 실제 전공명으로 업데이트
                break
            cm_keyword = cm_str.replace('전공', '').replace('과정', '')[:4]
            clean_keyword = clean_major.replace('전공', '').replace('과정', '')[:4]
            if cm_keyword == clean_keyword:
                courses = type_matched[type_matched['전공명'] == course_major]
                clean_major = cm_str
                break
    
    # 제도유형 표시용 정제
    display_program_type = program_type
    if is_micro:
        display_program_type = "소단위전공과정(마이크로디그리)"
    
    if not courses.empty:
        st.subheader(f"📚 ({display_program_type}) {clean_major} 편성 교과목(2025학년도 교육과정 기준) 안내")
        
        # 학년별 탭
        years = sorted([int(y) for y in courses['학년'].unique() if pd.notna(y)])
        
        if years:
            tabs = st.tabs([f"{year}학년" for year in years])
            
            for idx, year in enumerate(years):
                with tabs[idx]:
                    year_courses = courses[courses['학년'] == year]
                    semesters = sorted([int(s) for s in year_courses['학기'].unique() if pd.notna(s)])
                    
                    for semester in semesters:
                        st.markdown(f"#### 📅 {semester}학기")
                        
                        semester_courses = year_courses[year_courses['학기'] == semester]
                        
                        # 이수구분별 그룹화
                        required_courses = semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)]
                        elective_courses = semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)]
                        other_courses = semester_courses[
                            ~semester_courses['이수구분'].str.contains('필수', na=False) & 
                            ~semester_courses['이수구분'].str.contains('선택', na=False)
                        ]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if not required_courses.empty:
                                st.markdown("**🔴 전공필수**")
                                for _, row in required_courses.iterrows():
                                    course_name = row.get('과목명', '')
                                    credits = row.get('학점', '')
                                    try:
                                        credits_str = f"{int(credits)}학점"
                                    except:
                                        credits_str = ""
                                    st.write(f"• {course_name} ({credits_str})")
                        
                        with col2:
                            if not elective_courses.empty:
                                st.markdown("**🟢 전공선택**")
                                for _, row in elective_courses.iterrows():
                                    course_name = row.get('과목명', '')
                                    credits = row.get('학점', '')
                                    try:
                                        credits_str = f"{int(credits)}학점"
                                    except:
                                        credits_str = ""
                                    st.write(f"• {course_name} ({credits_str})")
                        
                        if not other_courses.empty:
                            st.markdown("**🔵 기타**")
                            for _, row in other_courses.iterrows():
                                division = row.get('이수구분', '')
                                course_name = row.get('과목명', '')
                                credits = row.get('학점', '')
                                try:
                                    credits_str = f"{int(credits)}학점"
                                except:
                                    credits_str = ""
                                st.write(f"• [{division}] {course_name} ({credits_str})")
                        
                        st.divider()
        
        # 전공 연락처 표시
        st.markdown("---")
        display_major_contact(clean_major)
        
        return True
    else:
        st.info(f"'{clean_major}' - '{display_program_type}'의 교과목 정보가 없습니다.")
        return False


def display_major_contact(major):
    """전공 연락처 표시"""
    if MAJORS_INFO.empty:
        return
    
    # 전공명 매칭
    contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == major]
    
    if contact_row.empty:
        # 부분 매칭 시도
        keyword = major.replace('전공', '').replace('과정', '')[:4]
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains(keyword, na=False)]
    
    if not contact_row.empty:
        row = contact_row.iloc[0]
        phone = row.get('연락처', '')
        location = row.get('사무실위치', '')
        
        contact_info = []
        if pd.notna(phone) and str(phone).strip():
            contact_info.append(f"📞 **연락처**: {phone}")
        if pd.notna(location) and str(location).strip():
            contact_info.append(f"📍 **사무실**: {location}")
        
        if contact_info:
            st.info("**📋 전공 문의처**\n\n" + "\n\n".join(contact_info))
        else:
            st.caption("📞 문의: 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다.")


# ============================================================
# 🖥️ 메인 UI
# ============================================================

def main():
    initialize_session_state()
    
    st.title(APP_TITLE)
    
    # === 사이드바 ===
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 10px 0;'>
                <h1 style='font-size: 3rem; margin-bottom: 0;'>🎓</h1>
                <h3 style='margin-top: 0;'>HKNU 다전공 안내</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내", "FAQ"], 
            icons=["chat-dots-fill", "journal-bookmark-fill", "question-circle-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#0091FF"},
            }
        )
        
        st.divider()
        
        with st.container(border=True):
            st.markdown("### 🤖 AI챗봇 안내")
            st.info("무엇이든 물어보세요!")
            # Semantic Router 상태 표시
            if SEMANTIC_ROUTER is not None:
                st.caption("🧠 Semantic Router 활성화")
            else:
                st.caption("⚡ 키워드 기반 분류 중")
            st.caption("* 정보는 참고용입니다.")
        
        st.markdown("---")
        st.caption("☏ 학사지원팀 031-670-5035")
        st.caption("* Powered by Gemini 2.0")
    
    # === 메인 콘텐츠 ===
    
    if menu == "AI챗봇 상담":
        st.subheader("💬 AI 상담원과 대화하기")
        
        # 예시 질문 버튼
        with st.expander("💡 어떤 질문을 해야 할지 모르겠나요? (클릭)", expanded=False):
            st.markdown("아래 버튼을 눌러 질문해보세요!")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📋 신청 관련", "📚 제도 안내", "🎓 학점/추천", "📞 전공/과목"])
            
            questions_by_tab = {
                "tab1": [
                    "신청 자격이 어떻게 되나요?",
                    "신청 기간은 언제인가요?",
                    "신청 방법 알려주세요",
                    "다전공 포기는 어떻게 하나요?"
                ],
                "tab2": [
                    "복수전공과 부전공 차이점",
                    "마이크로디그리가 뭐야?",
                    "융합전공 설명해줘",
                    "연계전공이 뭔가요?"
                ],
                "tab3": [
                    "제도별 이수 학점 알려줘",
                    "복수전공 하면 본전공 학점 변해?",
                    "2022학번 경영학전공, 전필3학점 전선9학점. 다전공 추천해줘",
                    "부전공은 몇 학점 이수해야 해?"
                ],
                "tab4": [
                    "경영학전공 연락처",
                    "소프트웨어융합전공 사무실 위치",
                    "AI반도체융합전공 어떤 과목 들어?",
                    "빅데이터융합전공 교과목 알려줘"
                ]
            }
            
            def handle_question_click(question):
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("AI가 답변을 생성 중입니다..."):
                    response_text, res_type = generate_ai_response(
                        question,
                        st.session_state.chat_history[:-1],
                        ALL_DATA
                    )
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response_text, 
                    "response_type": res_type
                })
                st.rerun()
            
            with tab1:
                cols = st.columns(2)
                for idx, q in enumerate(questions_by_tab["tab1"]):
                    if cols[idx % 2].button(f"💬 {q}", key=f"tab1_{idx}", use_container_width=True):
                        handle_question_click(q)
            
            with tab2:
                cols = st.columns(2)
                for idx, q in enumerate(questions_by_tab["tab2"]):
                    if cols[idx % 2].button(f"💬 {q}", key=f"tab2_{idx}", use_container_width=True):
                        handle_question_click(q)
            
            with tab3:
                cols = st.columns(2)
                for idx, q in enumerate(questions_by_tab["tab3"]):
                    if cols[idx % 2].button(f"💬 {q}", key=f"tab3_{idx}", use_container_width=True):
                        handle_question_click(q)
            
            with tab4:
                cols = st.columns(2)
                for idx, q in enumerate(questions_by_tab["tab4"]):
                    if cols[idx % 2].button(f"💬 {q}", key=f"tab4_{idx}", use_container_width=True):
                        handle_question_click(q)
        
        st.divider()
        
        # 채팅 기록 표시
        for chat in st.session_state.chat_history:
            role = "user" if chat["role"] == "user" else "assistant"
            avatar = "🧑‍🎓" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(chat["content"])
        
        # 입력창
        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI가 답변을 생성 중입니다..."):
                    response_text, res_type = generate_ai_response(
                        prompt, 
                        st.session_state.chat_history[:-1], 
                        ALL_DATA
                    )
                    st.markdown(response_text)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response_text, 
                "response_type": res_type
            })
            scroll_to_bottom()
    
    elif menu == "다전공 제도 안내":
        st.header("📊 제도 한눈에 비교")

        # 제도별 학점 정보를 실제 데이터에서 가져오는 함수
        def get_program_credits(program_name):
            """제도별 본전공/다전공 학점 정보 가져오기"""
            primary_credits = "-"
            multi_credits = "-"
            
            # 제도명 매핑 (UI 표시명 → 데이터 검색용)
            program_mapping = {
                '복수전공': '복수전공',
                '부전공': '부전공',
                '융합전공': '융합전공',
                '융합부전공': '융합부전공',
                '연계전공': '연계전공',
                '마이크로디그리': '소단위',
                '소단위전공과정': '소단위',
            }
            search_name = program_mapping.get(program_name, program_name)
            
            # 본전공 학점 (primary_requirements.xlsx에서)
            if 'primary_req' in ALL_DATA and not ALL_DATA['primary_req'].empty:
                primary_req = ALL_DATA['primary_req']
                filtered = primary_req[primary_req['제도유형'].str.contains(search_name, na=False)]
                if not filtered.empty:
                    row = filtered.sort_values('기준학번', ascending=False).iloc[0]
                    val = row.get('본전공_계', 0)
                    if pd.notna(val):
                        try:
                            primary_credits = f"{int(val)}학점"
                        except (ValueError, TypeError):
                            primary_credits = f"{val}학점"
            
            # 다전공 학점 (graduation_requirements.xlsx에서)
            if 'grad_req' in ALL_DATA and not ALL_DATA['grad_req'].empty:
                grad_req = ALL_DATA['grad_req']
                filtered = grad_req[grad_req['제도유형'].str.contains(search_name, na=False)]
                if not filtered.empty:
                    row = filtered.sort_values('기준학번', ascending=False).iloc[0]
                    val = row.get('다전공_계', 0)
                    if pd.notna(val):
                        try:
                            multi_credits = f"{int(val)}학점"
                        except (ValueError, TypeError):
                            multi_credits = f"{val}학점"
            
            return primary_credits, multi_credits

        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '설명 없음')
                    if pd.isna(desc) or desc == '':
                        desc = '설명 없음'
                    # programs.xlsx에서 학점 정보 직접 가져오기
                    c_pri = info.get('credits_primary', '-')
                    c_mul = info.get('credits_multi', '-')
                    if pd.isna(c_pri) or c_pri == '':
                        c_pri = '-'
                    if pd.isna(c_mul) or c_mul == '':
                        c_mul = '-'
                    degree = info.get('degree', '-')
                    if pd.isna(degree) or degree == '':
                        degree = '-'
                    difficulty = info.get('difficulty', '⭐')
                    if pd.isna(difficulty) or difficulty == '':
                        difficulty = '⭐⭐⭐'
                    
                    # 긴 텍스트 스타일
                    long_text_style = "overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; line-height: 1.4; font-size: 12px;"

                    html_content = f"""
                    <div style="border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); min-height: 420px; margin-bottom: 20px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h3 style="margin: 0 0 8px 0; color: #1f2937; font-size: 1.2rem;">🎓 {program}</h3>
                            <p style="color: #6b7280; font-size: 13px; margin-bottom: 12px;">{desc}</p>
                            <hr style="margin: 12px 0; border: 0; border-top: 1px solid #e5e7eb;">
                            <div style="font-size: 14px; margin-bottom: 8px;">
                                <strong style="color: #374151;">📖 이수 학점</strong>
                                <ul style="padding-left: 18px; margin: 4px 0; color: #4b5563; font-size: 12px;">
                                    <li style="margin-bottom: 4px;"><span style="font-weight:600; color:#374151;">본전공:</span> {c_pri}</li>
                                    <li><span style="font-weight:600; color:#374151;">다전공:</span> {c_mul}</li>
                                </ul>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: end; margin-top: 10px;">
                            <div style="max-width: 65%;">
                                <strong style="color: #374151; font-size: 14px;">📜 학위기</strong><br>
                                <div style="font-size: 12px; color: #2563eb; background: #eff6ff; padding: 2px 6px; border-radius: 4px; {long_text_style}">{degree}</div>
                            </div>
                            <div style="text-align: right; min-width: 30%;">
                                <strong style="color: #374151; font-size: 14px;">난이도</strong><br>
                                <span style="color: #f59e0b; font-size: 16px;">{difficulty}</span>
                            </div>
                        </div>
                    </div>"""
                    st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.error("❌ 제도 데이터를 불러오지 못했습니다.")

        st.divider()

        st.subheader("🔍 상세 정보 조회")
        
        prog_keys = list(ALL_DATA['programs'].keys()) if 'programs' in ALL_DATA else []
        selected_program = st.selectbox("자세히 알아볼 제도를 선택하세요", prog_keys)
        
        if selected_program and 'programs' in ALL_DATA:
            info = ALL_DATA['programs'][selected_program]
            
            # programs.xlsx에서 학점 정보 가져오기
            c_gen = info.get('credits_general', '-')
            c_pri = info.get('credits_primary', '-')
            c_mul = info.get('credits_multi', '-')
            if pd.isna(c_gen) or c_gen == '':
                c_gen = '-'
            if pd.isna(c_pri) or c_pri == '':
                c_pri = '-'
            if pd.isna(c_mul) or c_mul == '':
                c_mul = '-'
            
            tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징 및 유의사항"])
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**개요**\n\n{info.get('description', '-')}")
                    st.subheader("📖 이수 학점 상세")
                    st.markdown(f"""
- **교양:** {c_gen}
- **원전공:** {c_pri}
- **다전공:** {c_mul}
                    """)
                    st.subheader("🎓 졸업 요건")
                    st.markdown(f"- **졸업인증:** {info.get('graduation_certification', '-')}")
                    st.markdown(f"- **졸업시험:** {info.get('graduation_exam', '-')}")

                with col2:
                    st.success(f"**신청 자격**\n\n{info.get('qualification', '-')}")
                    st.write(f"**학위기 표기**\n\n{info.get('degree', '-')}")
            with tab2:
                for f in info.get('features', []): st.write(f"✔️ {f}")
                if info.get('notes'): st.warning(f"**💡 유의사항**: {info['notes']}")
            
            st.divider()

            # 전공명 -> 교육운영전공 매핑
            available_majors = {}
            
            # 🔥 정확한 제도유형 매칭 함수
            def match_program_type(type_value, selected_prog):
                type_str = str(type_value).strip()
                
                if "소단위" in selected_prog or "마이크로" in selected_prog:
                    return any(kw in type_str.lower() for kw in ['소단위', '마이크로', 'md'])
                
                if selected_prog == "부전공":
                    return "부전공" in type_str and "융합부전공" not in type_str
                
                if selected_prog == "융합전공":
                    return "융합전공" in type_str and "융합부전공" not in type_str
                
                return selected_prog in type_str
            
            # 전공명 정제 함수 (학부명 제거: "AI융합학부 AI빅데이터융합전공" -> "AI빅데이터융합전공")
            def clean_major_name(major_name):
                if not major_name or pd.isna(major_name):
                    return major_name
                name = str(major_name).strip()
                if ' ' in name:
                    parts = name.split(' ')
                    if len(parts) >= 2 and '학부' in parts[0]:
                        return ' '.join(parts[1:])
                return name
            
            if 'courses' in ALL_DATA and not ALL_DATA['courses'].empty:
                c_df = ALL_DATA['courses']
                if '제도유형' in c_df.columns:
                    mask = c_df['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                    for major in c_df[mask]['전공명'].unique():
                        cleaned = clean_major_name(major)
                        if cleaned not in available_majors:
                            available_majors[cleaned] = None

            if 'curriculum' in ALL_DATA:
                 curr_df = ALL_DATA['curriculum']
                 if not curr_df.empty and '제도유형' in curr_df.columns:
                     mask = curr_df['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                     for major in curr_df[mask]['전공명'].unique():
                         cleaned = clean_major_name(major)
                         if cleaned not in available_majors:
                             available_majors[cleaned] = None
            
            if 'majors' in ALL_DATA and not ALL_DATA['majors'].empty:
                m_df = ALL_DATA['majors']
                if '제도유형' in m_df.columns:
                    mask = m_df['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                    
                    for _, row in m_df[mask].iterrows():
                        major_name = clean_major_name(row['전공명'])
                        edu_major = row.get('교육운영전공', None)
                        
                        if pd.notna(edu_major) and str(edu_major).strip() not in ['', 'nan', '-']:
                            available_majors[major_name] = str(edu_major).strip()
                        elif major_name not in available_majors:
                            available_majors[major_name] = None
            
            # 중복 전공명 제거
            def remove_duplicate_majors(majors_dict):
                major_names = list(majors_dict.keys())
                to_remove = set()
                
                for i, name1 in enumerate(major_names):
                    for j, name2 in enumerate(major_names):
                        if i != j:
                            if name1 in name2 and len(name2) > len(name1):
                                to_remove.add(name2)
                
                for name in to_remove:
                    if name in majors_dict:
                        del majors_dict[name]
                
                return majors_dict
            
            available_majors = remove_duplicate_majors(available_majors)

            if available_majors:
                target_programs = ["복수전공", "부전공", "융합전공", "융합부전공"]
                
                if selected_program in target_programs:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors.keys())))
                    with col_m2:
                        all_majors_list = []
                        if 'primary_req' in ALL_DATA and not ALL_DATA['primary_req'].empty:
                            all_majors_list = sorted(ALL_DATA['primary_req']['전공명'].unique().tolist())
                        my_primary_major = st.selectbox("나의 본전공 (제1전공)", ["선택 안 함"] + all_majors_list)
                else:
                    if "소단위" in selected_program or "마이크로" in selected_program:
                        field_groups = {}
                        major_display_map = {}
                        
                        for major_name, edu_major in available_majors.items():
                            major_lower = str(major_name).lower()
                            
                            if edu_major:
                                display_name = f"{major_name} ({edu_major})"
                            else:
                                display_name = major_name
                            
                            major_display_map[display_name] = major_name
                            
                            if any(k in major_lower for k in ['식품', '농', '원예', '생명', '바이오']):
                                field = "🌾 농업·식품·바이오"
                            elif any(k in major_lower for k in ['디지털', 'ai', '인공지능', '데이터', '소프트웨어', 'ict', '스마트']):
                                field = "💻 ICT·디지털"
                            elif any(k in major_lower for k in ['경영', '창업', '마케팅', '금융', '회계']):
                                field = "💼 경영·창업"
                            elif any(k in major_lower for k in ['환경', '에너지', '기후']):
                                field = "🌍 환경·에너지"
                            elif any(k in major_lower for k in ['디자인', '미디어', '콘텐츠', '문화']):
                                field = "🎨 디자인·문화·콘텐츠"
                            elif any(k in major_lower for k in ['글로벌', '국제', '통상', '무역']):
                                field = "🌏 글로벌·국제"
                            elif any(k in major_lower for k in ['건강', '의료', '바이오헬스', '복지']):
                                field = "🏥 건강·의료"
                            else:
                                field = "📚 기타"
                            
                            if field not in field_groups:
                                field_groups[field] = []
                            field_groups[field].append(display_name)
                        
                        grouped_options = []
                        for field in sorted(field_groups.keys()):
                            grouped_options.append(f"━━━━━ {field} ━━━━━")
                            for display_name in sorted(field_groups[field]):
                                grouped_options.append(display_name)
                        
                        selected_option = st.selectbox(
                            f"이수하려는 {selected_program}", 
                            grouped_options,
                            help="분야별로 구분되어 있습니다"
                        )
                        
                        if selected_option.startswith("━━━━━"):
                            st.warning("⚠️ 구분선은 선택할 수 없습니다. 전공명을 선택해주세요.")
                            selected_major = None
                        else:
                            selected_major = major_display_map.get(selected_option, selected_option)
                    else:
                        selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors.keys())))
                    
                    my_primary_major = "선택 안 함"

                if selected_major:
                    if selected_program in target_programs:
                        current_year = datetime.now().year
                        admission_year = st.number_input(
                            "본인 학번 (입학연도)", 
                            min_value=2018, 
                            max_value=current_year, 
                            value=current_year
                        )
                        
                        st.write("")
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.subheader(f"🎯 {selected_program}({selected_major}) 이수 학점 기준")
                            
                            if 'grad_req' in ALL_DATA and not ALL_DATA['grad_req'].empty:
                                req_data = ALL_DATA['grad_req'][
                                    (ALL_DATA['grad_req']['전공명'] == selected_major) & 
                                    (ALL_DATA['grad_req']['제도유형'].str.contains(selected_program, na=False))
                                ].copy()
                                
                                req_data['기준학번'] = pd.to_numeric(req_data['기준학번'], errors='coerce')
                                req_data = req_data.dropna(subset=['기준학번'])
                                applicable = req_data[req_data['기준학번'] <= admission_year]
                                
                                if not applicable.empty:
                                    applicable = applicable.sort_values('기준학번', ascending=False)
                                    row = applicable.iloc[0]
                                    
                                    st.write(f"- 전공필수: **{int(row['다전공_전공필수'])}**학점")
                                    st.write(f"- 전공선택: **{int(row['다전공_전공선택'])}**학점")
                                    st.markdown(f"#### 👉 {selected_program} {int(row['다전공_계'])}학점")
                                else:
                                    st.warning(f"{admission_year}학번 기준 데이터가 없습니다.")
                            else:
                                st.warning("졸업요건 데이터가 없습니다.")

                        with col_right:
                            st.subheader(f"🏠 본전공({my_primary_major}) 이수 학점 기준")
                            
                            if my_primary_major != "선택 안 함" and 'primary_req' in ALL_DATA:
                                pri_data = ALL_DATA['primary_req'][ALL_DATA['primary_req']['전공명'] == my_primary_major].copy()
                                
                                if not pri_data.empty:
                                    pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
                                    pri_valid = pri_data[pri_data['기준학번'] <= admission_year]
                                    
                                    if not pri_valid.empty:
                                        matched_row = None
                                        pri_valid = pri_valid.sort_values('기준학번', ascending=False)
                                        
                                        for _, p_row in pri_valid.iterrows():
                                            if selected_program in str(p_row['제도유형']):
                                                matched_row = p_row
                                                break
                                        
                                        if matched_row is not None:
                                            st.write(f"- 본전공 전필: **{int(matched_row['본전공_전공필수'])}**학점")
                                            st.write(f"- 본전공 전선: **{int(matched_row['본전공_전공선택'])}**학점")
                                            st.markdown(f"#### 👉 본전공 {int(matched_row['본전공_계'])}학점으로 변경")
                                            
                                            if pd.notna(matched_row.get('비고')):
                                                st.caption(f"참고: {matched_row['비고']}")
                                        else:
                                            st.info(f"변동 데이터가 없습니다. (단일전공 기준 유지 가능성)")
                                    else:
                                        st.warning(f"{admission_year}학번 기준 데이터가 없습니다.")
                                else:
                                    st.warning("본전공 데이터를 찾을 수 없습니다.")
                            elif my_primary_major == "선택 안 함":
                                st.info("본전공을 선택하면 변동된 이수 학점을 확인할 수 있습니다.")

                    st.divider()

                    # 교과목 표시
                    if selected_program == "융합전공":
                        # 융합전공: 이수체계도 이미지 + 교과목 목록
                        st.subheader("📋 이수체계도")
                        display_curriculum_image(selected_major, selected_program)
            
                        if not COURSES_DATA.empty:
                            display_courses(selected_major, selected_program)
                    
                    elif "소단위" in selected_program or "마이크로" in selected_program:
                        # 소단위전공과정(마이크로디그리): 과정 안내 이미지 + 교과목 목록
                        st.subheader("🖼️ 과정 안내 이미지")
                        display_curriculum_image(selected_major, selected_program)
            
                        if not COURSES_DATA.empty:
                            display_courses(selected_major, selected_program)
                    
                    elif selected_program == "연계전공":
                        # 연계전공: 교과목 목록만
                        if not COURSES_DATA.empty:
                            display_courses(selected_major, selected_program)
                    
                    elif selected_program in ["복수전공", "부전공", "융합부전공"]:
                        # 복수전공/부전공/융합부전공: 교과목 목록만
                        if not COURSES_DATA.empty:
                            display_courses(selected_major, selected_program)

    elif menu == "FAQ":
        st.header("❓ 자주 묻는 질문")
        
        if FAQ_DATA:
            categories = list(set([faq.get('카테고리', '일반') for faq in FAQ_DATA]))
            categories = [c for c in categories if c and str(c).lower() not in ['nan', 'none', '']]
            
            if not categories:
                categories = ['일반']
            
            selected_category = st.selectbox("카테고리 선택", ["전체"] + sorted(categories))
            
            search_term = st.text_input("🔍 FAQ 검색", placeholder="키워드를 입력하세요...")
            
            filtered_faqs = FAQ_DATA
            
            if selected_category != "전체":
                filtered_faqs = [faq for faq in filtered_faqs if faq.get('카테고리') == selected_category]
            
            if search_term:
                search_lower = search_term.lower()
                filtered_faqs = [
                    faq for faq in filtered_faqs 
                    if search_lower in faq.get('질문', '').lower() or search_lower in faq.get('답변', '').lower()
                ]
            
            st.write(f"📋 총 {len(filtered_faqs)}개의 FAQ")
            st.divider()
            
            for faq in filtered_faqs:
                with st.expander(f"**Q. {faq.get('질문', '질문 없음')}**"):
                    st.markdown(f"**A.** {faq.get('답변', '답변 없음')}")
        else:
            st.warning("FAQ 데이터가 없습니다.")
        
        st.divider()
        st.info("💡 원하는 답변을 찾지 못하셨나요? **AI챗봇 상담**에서 직접 질문해보세요!")


# ============================================================
# 🚀 프로그램 실행
# ============================================================

if __name__ == "__main__":
    initialize_session_state()
    main()
