"""
============================================================
🎓 한경국립대학교 다전공 안내 AI챗봇
============================================================
버전: 3.6 (Modern UI 리디자인)
수정사항:
1. AI챗봇 과목 안내 - 학년/학기/이수구분별 정리
2. 소단위전공 이미지 2개 표시 문제 해결
3. 소단위전공 교과목 'XX MD' 패턴으로 검색
4. 전공 문의처에 전공명, 위치 추가
5. 제도 비교 카드에 졸업요건, 신청자격 추가
6. 모바일 Streamlit 브랜딩 완전 숨김
7. 모바일 가독성 개선 (줄넘김 방지)
8. 임베딩 모델 업그레이드 (KoSimCSE)
9. "다전공이 뭐야" 질문 처리 개선
10. 과목 안내 시 학사공지 교육과정 참고 안내 추가
11. HTML 카드 스타일 UI 적용
12. 사이드바 AI챗봇/다전공 소개 스타일링
13. 질문 버튼 전체 그리드 방식 (24개 항목)
14. 계열별 전공 그룹화 (다전공 제도 안내 + AI챗봇)
15. Modern UI 전면 리디자인 ← 🆕
    - Pretendard 폰트 적용
    - 인디고(#4F46E5) 색상 팔레트
    - 부드러운 그림자 & 둥근 모서리
    - 채팅 메시지 스타일링 개선
    - 버튼 호버 효과
============================================================
"""

import streamlit as st
from google import genai
import pandas as pd
from streamlit_option_menu import option_menu 
from datetime import datetime
import os
import yaml
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

MESSAGES = load_yaml_config('messages.yaml')
MAPPINGS = load_yaml_config('mappings.yaml')
SETTINGS = load_yaml_config('settings.yaml')

# ============================================================
# 📌 상수 정의
# ============================================================

DEFAULT_CONTACT_MESSAGE = "📞 문의: 전공 사무실 또는 학사지원팀 031-670-5035로 연락주시면 보다 상세한 정보를 안내 받을 수 있습니다."
CONTACT_MESSAGE = MESSAGES.get('contact', {}).get('default', DEFAULT_CONTACT_MESSAGE)

APP_PERIOD = MESSAGES.get('application_period', {})
APP_PERIOD_TITLE = APP_PERIOD.get('title', "📅 다전공 신청 기간 안내")
APP_PERIOD_INTRO = APP_PERIOD.get('intro', "다전공 신청은 **매 학기 2회** 진행됩니다.")
APP_PERIOD_1ST = APP_PERIOD.get('first_semester', "전학기 **10월** / **12월**")
APP_PERIOD_2ND = APP_PERIOD.get('second_semester', "전학기 **4월** / **6월**")

LINKS = MESSAGES.get('links', {})
ACADEMIC_NOTICE_URL = LINKS.get('academic_notice', "https://www.hknu.ac.kr/kor/562/subview.do")

PATHS = SETTINGS.get('paths', {})
CURRICULUM_IMAGES_PATH = PATHS.get('curriculum_images', "images/curriculum")

APP_CONFIG = SETTINGS.get('app', {})
APP_TITLE = APP_CONFIG.get('title', "🎓 한경국립대 다전공 안내")

DIFFICULTY_STARS = MAPPINGS.get('difficulty_stars', {})

def convert_difficulty_to_stars(value):
    if pd.isna(value) or value == '':
        return DIFFICULTY_STARS.get('default', '⭐⭐⭐')
    if isinstance(value, str) and '⭐' in value:
        return value
    try:
        num = int(float(value))
        return DIFFICULTY_STARS.get(num, DIFFICULTY_STARS.get('default', '⭐⭐⭐'))
    except:
        return DIFFICULTY_STARS.get('default', '⭐⭐⭐')

# Semantic Router 설정
logging.getLogger("semantic_router").setLevel(logging.ERROR)
SEMANTIC_ROUTER_ENABLED = True

SEMANTIC_ROUTER_AVAILABLE = False
Route = None
SemanticRouter = None
HuggingFaceEncoder = None
LocalIndex = None

try:
    from semantic_router import Route
    from semantic_router.routers import SemanticRouter
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_router.index import LocalIndex
    SEMANTIC_ROUTER_AVAILABLE = True
    SEMANTIC_ROUTER_VERSION = "0.1.x"
except ImportError:
    try:
        from semantic_router import Route
        from semantic_router.layer import RouteLayer as SemanticRouter
        from semantic_router.encoders import HuggingFaceEncoder
        SEMANTIC_ROUTER_AVAILABLE = True
        SEMANTIC_ROUTER_VERSION = "0.0.x"
    except ImportError:
        SEMANTIC_ROUTER_AVAILABLE = False
        SEMANTIC_ROUTER_VERSION = None

# Gemini API 설정
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# 페이지 설정
st.set_page_config(
    page_title="다전공 안내 AI챗봇",
    page_icon="🎓",
    layout="wide",
)

# 🔧 수정 #6, #7: CSS - Modern UI 스타일링
modern_css = """
<style>
/* 폰트 적용 (Pretendard) */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* 전체 배경 */
.stApp {
    background-color: #F8F9FC;
}

/* 헤더/푸터 숨김 */
header {visibility: hidden !important;}
footer {display: none !important; visibility: hidden !important; height: 0 !important;}
.stApp > footer {display: none !important;}
#MainMenu {visibility: hidden !important;}
[data-testid="stToolbar"] {display: none !important;}
.stDeployButton {display: none !important;}
a[href*="streamlit.io"] {display: none !important;}

/* 메인 컨테이너 */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 8rem !important;
    max-width: 1000px;
}

/* 사이드바 토글 버튼 유지 */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: block !important;
}

/* 채팅 메시지 스타일링 */
[data-testid="stChatMessage"] {
    background-color: transparent;
    padding: 1rem 0;
}
[data-testid="stChatMessage"] .stMarkdown {
    background-color: #ffffff;
    padding: 16px 20px;
    border-radius: 0px 20px 20px 20px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    border: 1px solid #E5E7EB;
    line-height: 1.6;
}
[data-testid="chatAvatarIcon-user"] {
    background-color: #4F46E5 !important;
}

/* 버튼 스타일링 */
.stButton > button {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    background-color: white !important;
    color: #374151 !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    height: auto !important;
}
.stButton > button:hover {
    border-color: #4F46E5 !important;
    color: #4F46E5 !important;
    background-color: #EEF2FF !important;
    transform: translateY(-1px);
}

/* 입력창 스타일 */
.stChatInputContainer {
    position: sticky;
    bottom: 0;
    background: #F8F9FC;
    padding: 1rem 0;
    z-index: 999;
}
.stChatInputContainer textarea {
    border-radius: 24px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}

/* 탭 스타일 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    background-color: white;
    border: 1px solid #E5E7EB;
    padding: 0 16px;
    font-size: 14px;
}
.stTabs [aria-selected="true"] {
    background-color: #4F46E5 !important;
    color: white !important;
    border: none !important;
}

/* 사이드바 스타일 */
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #F3F4F6;
}

/* 테이블 스타일 */
table {
    border-collapse: separate !important; 
    border-spacing: 0;
    width: 100%;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    overflow: hidden;
}
th {
    background-color: #F9FAFB !important;
    color: #4B5563 !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #E5E7EB !important;
    padding: 12px !important;
}
td {
    padding: 12px !important;
    border-bottom: 1px solid #F3F4F6 !important;
    font-size: 0.95rem;
}

/* Expander 스타일 */
.streamlit-expanderHeader {
    background-color: white !important;
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
}

/* 모바일 최적화 */
@media (max-width: 768px) {
    .main .block-container { 
        padding: 1rem 0.5rem !important; 
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    
    .stButton > button {
        font-size: 13px !important;
        padding: 8px 12px !important;
    }
    
    section[data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 250px !important;
    }
}

@media (max-width: 375px) {
    h1, h2 { font-size: 1rem !important; }
}

html, body {
    scroll-behavior: smooth;
}
</style>
"""
st.markdown(modern_css, unsafe_allow_html=True)


def scroll_to_bottom():
    js = """
    <script>
        setTimeout(function() {
            var messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length > 0) {
                messages[messages.length - 1].scrollIntoView({behavior: "smooth", block: "end"});
            }
        }, 300);
    </script>
    """
    st.components.v1.html(js, height=0)


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
    try:
        if os.path.exists(file_path):
            result = pd.read_excel(file_path, sheet_name=sheet_name)
            if isinstance(result, dict):
                return list(result.values())[0] if result else pd.DataFrame()
            return result
        return pd.DataFrame()
    except Exception as e:
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
# 🧠 Semantic Router 설정
# ============================================================

INTENT_UTTERANCES = {
    'QUALIFICATION': [
        "신청 자격이 어떻게 되나요?", "지원 자격 알려주세요", "누가 신청할 수 있어요?",
        "자격 요건이 뭐예요?", "나도 신청 가능해?", "몇 학년부터 할 수 있어요?",
        "2학년인데 가능한가요?", "학점이 낮아도 되나요?", "조건이 어떻게 돼?",
        "신청 조건 알려줘", "자격이 되는지 모르겠어", "나 자격 있어?",
    ],
    'APPLICATION_PERIOD': [
        "신청 기간이 언제예요?", "언제 신청해요?", "마감일이 언제야?",
        "지원 기간 알려주세요", "언제까지 신청할 수 있어요?", "접수 기간이 어떻게 돼?",
        "몇 월에 신청해?", "신청 시작일이 언제야?", "지금 신청 가능해?",
    ],
    'APPLICATION_METHOD': [
        "신청 방법이 어떻게 되나요?", "어떻게 신청해요?", "신청 절차 알려주세요",
        "지원하려면 어떻게 해야 해?", "신청하는 법 알려줘", "어디서 신청해?",
        "절차가 어떻게 돼?", "지원 방법이 뭐야?",
    ],
    'CANCEL': [
        "포기하고 싶어요", "취소 방법 알려주세요", "철회하려면 어떻게 해?",
        "그만두고 싶어", "포기 신청 어떻게 해?", "취소할 수 있어?",
        "다전공 포기", "복수전공 취소", "포기 방법",
    ],
    'CHANGE': [
        "변경하고 싶어요", "전공 바꾸고 싶어", "수정할 수 있나요?",
        "전환하려면 어떻게 해?", "복수전공에서 부전공으로 바꾸고 싶어",
        "변경 가능한가요?", "전공 변경 방법", "바꿀 수 있어?",
    ],
    'PROGRAM_COMPARISON': [
        "복수전공이랑 부전공 차이가 뭐야?", "뭐가 다른 거야?", "차이점 알려줘",
        "비교해줘", "뭐가 더 좋아?", "어떤 게 나을까?",
        "융합전공이랑 복수전공 비교", "차이점이 뭐예요?", "장단점 비교",
    ],
    'CREDIT_INFO': [
        "학점이 몇 학점이야?", "이수 학점 알려줘", "졸업하려면 몇 학점 필요해?",
        "본전공 학점이 줄어들어?", "학점 변화 알려줘", "총 학점이 어떻게 돼?",
        "전필 몇 학점이야?", "전선 학점은?", "필요한 학점 수",
    ],
    'PROGRAM_INFO': [
        "복수전공이 뭐야?", "부전공이 뭔가요?", "융합전공 설명해줘",
        "마이크로디그리가 뭐예요?", "연계전공이 뭐지?", "이게 뭐야?",
        "알려줘", "설명해줘", "무슨 제도야?", "소단위전공이 뭐야?",
        "다전공이 뭐야?", "다전공 제도가 뭐야?", "유연학사제도가 뭐야?",  # 🔧 수정 #9
    ],
    'COURSE_SEARCH': [
        "어떤 과목 들어야 해?", "커리큘럼 알려줘", "수업 뭐 들어?",
        "과목 리스트 보여줘", "뭐 배워?", "교과목 알려줘",
        "강의 뭐 있어?", "필수 과목이 뭐야?", "어떤 강의 들어야 해?",
    ],
    'CONTACT_SEARCH': [
        "연락처 알려줘", "전화번호가 뭐야?", "문의 어디로 해?",
        "사무실 어디야?", "담당자 연락처", "위치가 어디야?",
    ],
    'RECOMMENDATION': [
        "뭐가 좋을까?", "추천해줘", "어떤 게 좋아?", "나한테 맞는 거 뭐야?",
        "뭐 해야 할까?", "어떤 걸 선택해야 할까?", "추천 좀 해줘",
        "뭐가 유리할까?", "골라줘", "선택 도와줘",
    ],
    'GREETING': [
        "안녕", "안녕하세요", "하이", "hello", "hi", "반가워",
    ],
    'OUT_OF_SCOPE': [
        "오늘 날씨 어때?", "맛집 추천해줘", "취업 어떻게 해?",
        "기숙사 신청 어떻게 해?", "장학금 어떻게 받아?", "수강신청 어떻게 해?",
        "휴학 신청 방법", "교환학생 어떻게 가?", "너 누구야?",
    ],
    'BLOCKED': [
        "시발", "씨발", "ㅅㅂ", "병신", "ㅂㅅ", "지랄", "ㅈㄹ",
        "개새끼", "꺼져", "닥쳐", "죽어", "미친", "존나", "fuck",
    ],
}

INTENT_KEYWORDS = {
    'QUALIFICATION': ['신청자격', '지원자격', '자격요건', '자격이뭐', '누가신청', '신청조건'],
    'APPLICATION_PERIOD': ['신청기간', '지원기간', '접수기간', '언제신청', '마감일', '언제까지'],
    'APPLICATION_METHOD': ['신청방법', '지원방법', '신청절차', '어떻게신청', '어디서신청'],
    'CANCEL': ['포기', '취소', '철회', '그만', '중단'],
    'CHANGE': ['변경', '수정', '바꾸', '전환'],
    'PROGRAM_COMPARISON': ['차이', '비교', 'vs', '다른점', '뭐가달라'],
    'CREDIT_INFO': ['학점', '이수학점', '졸업요건', '몇학점', '학점변화'],
    'PROGRAM_INFO': ['뭐야', '무엇', '뭔가요', '알려줘', '설명'],
    'COURSE_SEARCH': ['과목', '수업', '강의', '커리큘럼', '교과목'],
    'CONTACT_SEARCH': ['연락처', '전화번호', '문의', '사무실', '위치'],
    'RECOMMENDATION': ['추천', '뭐할까', '선택', '고민', '좋을까'],
    'GREETING': ['안녕', '하이', 'hello', 'hi', '반가'],
    'OUT_OF_SCOPE': ['날씨', '맛집', '취업', '기숙사', '장학금', '수강신청', '휴학'],
    'BLOCKED': ['시발', '씨발', 'ㅅㅂ', '병신', 'ㅂㅅ', '지랄', '개새끼', '존나', 'fuck'],
}

PROGRAM_KEYWORDS = {
    '복수전공': ['복수전공', '복전', '복수'],
    '부전공': ['부전공', '부전'],
    '융합전공': ['융합전공', '융합'],
    '융합부전공': ['융합부전공'],
    '연계전공': ['연계전공', '연계'],
    '마이크로디그리': ['마이크로디그리', '마이크로', 'md', '소단위전공과정', '소단위전공', '소단위'],
}


@st.cache_resource
def initialize_semantic_router():
    if not SEMANTIC_ROUTER_AVAILABLE or not SEMANTIC_ROUTER_ENABLED:
        return None
    if Route is None or SemanticRouter is None or HuggingFaceEncoder is None:
        return None
    try:
        # 🔧 임베딩 모델 업그레이드: 축약어, 구어체, 모호한 질문 처리 향상
        encoder = HuggingFaceEncoder(name="BM-K/KoSimCSE-roberta-multitask")
        routes = [Route(name=intent_name, utterances=utterances) 
                  for intent_name, utterances in INTENT_UTTERANCES.items()]
        if LocalIndex is not None:
            router = SemanticRouter(encoder=encoder, routes=routes, index=LocalIndex())
        else:
            router = SemanticRouter(encoder=encoder, routes=routes)
        return router
    except Exception as e:
        return None


SEMANTIC_ROUTER = initialize_semantic_router()


# ============================================================
# 🔥 의도 분류 함수
# ============================================================

def extract_programs(text):
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
    info = {}
    user_clean = user_input.lower().replace(' ', '')
    
    found_programs = extract_programs(user_clean)
    if found_programs:
        info['programs'] = found_programs
        info['program'] = found_programs[0]
    
    year_match = re.search(r'(20\d{2})', user_input)
    if year_match:
        info['year'] = int(year_match.group(1))
    
    credit_match = re.search(r'(\d+)\s*학점', user_input)
    if credit_match:
        info['credits'] = int(credit_match.group(1))
    
    major_patterns = [r'([가-힣A-Za-z]+(?:융합)?전공)', r'([가-힣A-Za-z]+학과)']
    for pattern in major_patterns:
        major_match = re.search(pattern, user_input)
        if major_match:
            major_name = major_match.group(1)
            if major_name not in ['복수전공', '부전공', '융합전공', '융합부전공', '연계전공', '다전공']:
                info['major'] = major_name
                break
    
    return info


def classify_with_semantic_router(user_input):
    if SEMANTIC_ROUTER is None:
        return None, 0.0
    try:
        result = SEMANTIC_ROUTER(user_input)
        if result and result.name:
            return result.name, 0.8
        return None, 0.0
    except:
        return None, 0.0


def classify_with_keywords(user_input):
    user_clean = user_input.lower().replace(' ', '')
    priority_order = [
        'QUALIFICATION', 'APPLICATION_PERIOD', 'APPLICATION_METHOD',
        'CANCEL', 'CHANGE', 'PROGRAM_COMPARISON', 'RECOMMENDATION',
        'CREDIT_INFO', 'PROGRAM_INFO', 'COURSE_SEARCH', 'CONTACT_SEARCH', 'GREETING',
    ]
    for intent in priority_order:
        keywords = INTENT_KEYWORDS.get(intent, [])
        if any(kw in user_clean for kw in keywords):
            return intent
    return None


def classify_with_ai(user_input):
    prompt = """당신은 질문 분류 AI입니다. 의도를 분류하세요.
[의도]: QUALIFICATION, APPLICATION_PERIOD, APPLICATION_METHOD, CANCEL, CHANGE, 
PROGRAM_COMPARISON, PROGRAM_INFO, CREDIT_INFO, COURSE_SEARCH, CONTACT_SEARCH, 
RECOMMENDATION, GREETING, OUT_OF_SCOPE
규칙: 의도 이름만 출력. "다전공이 뭐야?"는 PROGRAM_INFO"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"질문: {user_input}\n\n의도를 분류하세요.",
            config={'system_instruction': prompt, 'temperature': 0, 'max_output_tokens': 50}
        )
        intent = response.text.strip().upper()
        valid_intents = ['QUALIFICATION', 'APPLICATION_PERIOD', 'APPLICATION_METHOD',
                         'CANCEL', 'CHANGE', 'PROGRAM_COMPARISON', 'PROGRAM_INFO',
                         'CREDIT_INFO', 'COURSE_SEARCH', 'CONTACT_SEARCH',
                         'RECOMMENDATION', 'GREETING', 'OUT_OF_SCOPE']
        for valid in valid_intents:
            if valid in intent:
                return valid
        return 'OUT_OF_SCOPE'
    except:
        return 'OUT_OF_SCOPE'


def classify_intent(user_input, use_ai_fallback=True):
    """의도 분류 - 8가지 수정사항 반영"""
    user_clean = user_input.lower().replace(' ', '')
    
    # 🚫 욕설 차단
    if any(kw in user_clean for kw in INTENT_KEYWORDS.get('BLOCKED', [])):
        return 'BLOCKED', 'blocked', {}
    
    # 🔧 수정 #9: "다전공이 뭐야?" 우선 처리
    if '다전공' in user_clean and any(kw in user_clean for kw in ['뭐', '무엇', '알려', '설명', '뭔가', '뭐야']):
        if not any(prog in user_clean for prog in ['복수전공', '부전공', '융합전공', '융합부전공', '연계전공', '마이크로']):
            return 'PROGRAM_INFO', 'complex', {'program': '다전공'}
    
    # 복합 조건 검사
    has_course_keyword = any(kw in user_clean for kw in ['교과목', '과목', '커리큘럼', '수업'])
    has_major = bool(re.search(r'([가-힣]+(?:학|공학|과학|전공))', user_clean))
    
    if has_course_keyword and has_major:
        return 'COURSE_SEARCH', 'complex', extract_additional_info(user_input, 'COURSE_SEARCH')
    
    found_programs = extract_programs(user_clean)
    
    if found_programs:
        program = found_programs[0]
        if any(kw in user_clean for kw in ['자격', '신청할수있', '조건']):
            return 'QUALIFICATION', 'complex', {'program': program, 'programs': found_programs}
        if any(kw in user_clean for kw in ['언제', '기간', '마감']):
            return 'APPLICATION_PERIOD', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['어떻게', '방법', '절차']):
            return 'APPLICATION_METHOD', 'complex', {'program': program}
    
    # Semantic Router
    if SEMANTIC_ROUTER is not None:
        semantic_intent, score = classify_with_semantic_router(user_input)
        if semantic_intent:
            return semantic_intent, 'semantic', extract_additional_info(user_input, semantic_intent)
    
    # 키워드 분류
    keyword_intent = classify_with_keywords(user_input)
    if keyword_intent:
        return keyword_intent, 'keyword', extract_additional_info(user_input, keyword_intent)
    
    # 제도 설명 질문
    if found_programs:
        if any(kw in user_clean for kw in ['뭐', '무엇', '알려', '설명']):
            return 'PROGRAM_INFO', 'keyword', {'program': found_programs[0]}
    
    # AI 분류
    if use_ai_fallback:
        try:
            ai_intent = classify_with_ai(user_input)
            if ai_intent != 'GENERAL':
                return ai_intent, 'ai', extract_additional_info(user_input, ai_intent)
        except:
            pass
    
    return 'OUT_OF_SCOPE', 'fallback', {}

# ============================================================
# 🏫 계열별 전공 그룹화 헬퍼 함수
# ============================================================

def get_majors_by_category(program_type=None, data_source="majors"):
    """
    계열별로 전공을 그룹화하여 반환
    - 융합전공, 융합부전공, 소단위전공과정은 계열 구분 없이 반환
    - 일반 전공(복수전공, 부전공)은 계열별로 그룹화
    
    Returns:
        dict: {'계열명': ['전공1', '전공2', ...], ...}
        특수 제도의 경우: {'전체': ['전공1', '전공2', ...]}
    """
    # 특수 제도는 계열 구분 없음
    special_programs = ["융합전공", "융합부전공", "소단위전공과정", "연계전공"]
    
    if program_type in special_programs:
        majors_list = []
        
        if data_source == "majors" and not MAJORS_INFO.empty and '제도유형' in MAJORS_INFO.columns:
            if program_type == "융합전공":
                mask = MAJORS_INFO['제도유형'].str.contains('융합전공', na=False) & ~MAJORS_INFO['제도유형'].str.contains('융합부전공', na=False)
            elif "소단위" in program_type:
                mask = MAJORS_INFO['제도유형'].apply(lambda x: any(kw in str(x).lower() for kw in ['소단위', '마이크로', 'md']))
            else:
                mask = MAJORS_INFO['제도유형'].str.contains(program_type, na=False)
            majors_list = MAJORS_INFO[mask]['전공명'].unique().tolist()
        
        if data_source == "courses" and not COURSES_DATA.empty and '제도유형' in COURSES_DATA.columns:
            if program_type == "융합전공":
                mask = COURSES_DATA['제도유형'].str.contains('융합전공', na=False) & ~COURSES_DATA['제도유형'].str.contains('융합부전공', na=False)
            elif "소단위" in program_type:
                mask = COURSES_DATA['제도유형'].apply(lambda x: any(kw in str(x).lower() for kw in ['소단위', '마이크로', 'md']))
            else:
                mask = COURSES_DATA['제도유형'].str.contains(program_type, na=False)
            for m in COURSES_DATA[mask]['전공명'].unique():
                if m not in majors_list:
                    majors_list.append(m)
        
        return {"전체": sorted(majors_list)} if majors_list else {}
    
    # 일반 전공 (복수전공, 부전공) - 계열별 그룹화
    category_majors = {}
    
    if not MAJORS_INFO.empty:
        has_category = '계열' in MAJORS_INFO.columns
        
        if program_type:
            if program_type == "부전공":
                mask = MAJORS_INFO['제도유형'].str.contains('부전공', na=False) & ~MAJORS_INFO['제도유형'].str.contains('융합부전공', na=False)
            else:
                mask = MAJORS_INFO['제도유형'].str.contains(program_type, na=False)
            filtered_df = MAJORS_INFO[mask]
        else:
            filtered_df = MAJORS_INFO
        
        if has_category:
            for _, row in filtered_df.iterrows():
                category = row.get('계열', '기타')
                if pd.isna(category) or str(category).strip() == '':
                    category = '기타'
                category = str(category).strip()
                major_name = row['전공명']
                
                if category not in category_majors:
                    category_majors[category] = []
                if major_name not in category_majors[category]:
                    category_majors[category].append(major_name)
        else:
            # 계열 컬럼이 없으면 전체로 그룹화
            category_majors["전체"] = filtered_df['전공명'].unique().tolist()
    
    # 각 계열 내 전공 정렬
    for cat in category_majors:
        category_majors[cat] = sorted(category_majors[cat])
    
    return category_majors


def get_category_color(category):
    """계열별 색상 반환 - Modern 팔레트"""
    colors = {
        '공학계열': '#EF4444',      # Red
        '자연과학계열': '#10B981',   # Emerald
        '인문사회계열': '#3B82F6',   # Blue
        '예체능계열': '#8B5CF6',     # Violet
        '의학계열': '#F59E0B',       # Amber
        '사범계열': '#06B6D4',       # Cyan
        '기타': '#6B7280',           # Gray
        '전체': '#4F46E5',           # Indigo
    }
    return colors.get(category, '#6B7280')


def format_majors_by_category_html(category_majors):
    """계열별 전공 목록을 Modern HTML 카드로 포맷팅"""
    if not category_majors:
        return "<p style='color: #6B7280;'>전공 정보가 없습니다.</p>"
    
    html = ""
    for category, majors in category_majors.items():
        if not majors:
            continue
        color = get_category_color(category)
        majors_tags = " ".join([f'<span style="background: {color}15; color: {color}; padding: 4px 10px; border-radius: 20px; font-size: 13px; margin: 4px; display: inline-block; font-weight: 500;">{m}</span>' for m in majors])
        
        html += f"""
<div style="margin-bottom: 16px;">
    <div style="color: {color}; font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
        <span style="width: 8px; height: 8px; background: {color}; border-radius: 50%; display: inline-block;"></span>
        {category} ({len(majors)})
    </div>
    <div style="background: white; padding: 12px; border-radius: 12px; border: 1px solid #E5E7EB;">
        {majors_tags}
    </div>
</div>
"""
    return html


# ============================================================
# 🎨 Modern UI 카드 스타일 헬퍼 함수들
# ============================================================

def create_header_card(title, emoji="📋", gradient=None):
    """깔끔한 Modern 헤더 카드"""
    return f"""
<div style="background-color: white; border-bottom: 2px solid #4F46E5; padding: 20px 0; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="background-color: #EEF2FF; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
            {emoji}
        </div>
        <h3 style="margin: 0; color: #111827; font-weight: 700; font-size: 1.2rem;">{title}</h3>
    </div>
</div>
"""

def create_info_card(title, content_list, color="#4F46E5", emoji="📌"):
    """Modern 정보 카드 (Soft Shadow)"""
    items_html = "".join([f'<li style="margin-bottom: 6px; color: #374151;">{item}</li>' for item in content_list])
    
    return f"""
<div style="background: white; border-radius: 16px; padding: 20px; margin: 12px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #F3F4F6;">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="color: {color}; font-size: 1.1rem;">{emoji}</span>
        <strong style="color: #1F2937; font-size: 1rem;">{title}</strong>
    </div>
    <ul style="margin: 0; padding-left: 20px; font-size: 0.95rem; line-height: 1.6;">
        {items_html}
    </ul>
</div>
"""

def create_simple_card(content, bg_color="#F9FAFB", border_color="#E5E7EB"):
    """간결한 메시지 박스"""
    return f"""
<div style="background: {bg_color}; border: 1px solid {border_color}; padding: 16px; margin: 10px 0; border-radius: 12px; color: #374151;">
    {content}
</div>
"""

def create_step_card(step_num, title, description, color="#4F46E5"):
    """단계별 카드 (타임라인 스타일)"""
    return f"""
<div style="display: flex; gap: 16px; margin-bottom: 16px; align-items: flex-start;">
    <div style="background: {color}; color: white; min-width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; margin-top: 2px;">{step_num}</div>
    <div style="background: white; padding: 16px; border-radius: 12px; border: 1px solid #E5E7EB; flex-grow: 1; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);">
        <strong style="display: block; color: #111827; margin-bottom: 4px;">{title}</strong>
        <span style="color: #6B7280; font-size: 0.9rem;">{description}</span>
    </div>
</div>
"""

def create_tip_box(text, emoji="💡"):
    """팁 박스 - 앰버 색상"""
    return f"""
<div style="background: #FFFBEB; border: 1px solid #FCD34D; padding: 16px; margin: 16px 0; border-radius: 12px; display: flex; gap: 12px; align-items: center;">
    <span style="font-size: 1.2rem;">{emoji}</span>
    <span style="color: #92400E; font-size: 0.9rem; font-weight: 500;">{text}</span>
</div>
"""

def create_warning_box(text, emoji="⚠️"):
    """경고 박스 - 레드 색상"""
    return f"""
<div style="background: #FEF2F2; border: 1px solid #FECACA; padding: 16px; margin: 16px 0; border-radius: 12px; display: flex; gap: 12px; align-items: center;">
    <span style="font-size: 1.2rem;">{emoji}</span>
    <span style="color: #991B1B; font-size: 0.9rem; font-weight: 500;">{text}</span>
</div>
"""

def create_contact_box():
    """연락처 박스 - 깔끔한 스타일"""
    return f"""
<div style="margin-top: 24px; padding: 16px; background: white; border-radius: 12px; border: 1px solid #E5E7EB; text-align: center;">
    <p style="margin: 0; color: #6B7280; font-size: 0.9rem;">
        📞 문의가 필요하신가요?<br>
        <strong style="color: #4F46E5; font-size: 1rem;">전공 사무실</strong> 또는 <strong style="color: #4F46E5;">학사지원팀 031-670-5035</strong>
    </p>
</div>
"""

def create_table_html(headers, rows, colors=None):
    """Clean Table Design"""
    if colors is None:
        colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
    
    header_html = "".join([f'<th style="padding: 12px 16px; text-align: left; font-weight: 600;">{h}</th>' for h in headers])
    
    rows_html = ""
    for idx, row in enumerate(rows):
        cells = ""
        for i, cell in enumerate(row):
            if i == 0:
                color = colors[idx % len(colors)]
                cells += f'<td style="padding: 12px 16px;"><span style="color: {color}; font-weight: 600;">●</span> {cell}</td>'
            else:
                cells += f'<td style="padding: 12px 16px; color: #374151;">{cell}</td>'
        rows_html += f"<tr style='border-bottom: 1px solid #F3F4F6;'>{cells}</tr>\n"
    
    return f"""
<div style="overflow-x: auto; margin: 16px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border-radius: 12px; border: 1px solid #E5E7EB;">
    <table style="width: 100%; border-collapse: collapse; background: white;">
        <thead style="background: #F9FAFB; border-bottom: 1px solid #E5E7EB;">
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</div>
"""

def create_program_badge(program_name, color="#4F46E5"):
    """프로그램 배지 생성"""
    return f'<span style="background: {color}15; color: {color}; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-right: 6px;">{program_name}</span>'


# ============================================================
# 🎯 핸들러 함수들
# ============================================================

def handle_qualification(user_input, extracted_info, data_dict):
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    response = create_header_card("제도별 신청 자격", "📋")
    
    # Modern 색상 팔레트
    colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
    for idx, (p_name, p_info) in enumerate(programs.items()):
        qual = p_info.get('qualification', '-')
        color = colors[idx % len(colors)]
        response += create_info_card(p_name, [qual], color, "🎓")
    
    response += create_tip_box("학점이 부족하면 마이크로디그리부터 시작해보세요!")
    response += create_contact_box()
    
    return response, "QUALIFICATION"


def handle_application_period(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 신청 기간", "📅")
    
    response += create_simple_card("<p style='margin:0; text-align:center; font-weight:600; color: #111827;'>매 학기 2회 (4월/6월, 10월/12월)</p>")
    
    # 테이블
    headers = ["이수 희망 학기", "신청 시기"]
    rows = [
        ["1학기 이수 희망", f"{APP_PERIOD_1ST}"],
        ["2학기 이수 희망", f"{APP_PERIOD_2ND}"]
    ]
    response += create_table_html(headers, rows, ["#28a745", "#17a2b8"])
    
    response += create_warning_box(f'정확한 일정은 <a href="{ACADEMIC_NOTICE_URL}" style="color: #dc3545;">학사공지</a>를 반드시 확인하세요!')
    response += create_contact_box()
    
    return response, "APPLICATION_PERIOD"


def handle_application_method(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 신청 방법", "📝", "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)")
    
    response += create_step_card(1, "신청 시기 확인", "학사 공지사항에서 신청 기간을 확인합니다.", "#f5576c")
    response += create_step_card(2, "자격 요건 확인", "본인의 학년, 평점 등 자격 충족 여부를 확인합니다.", "#f093fb")
    response += create_step_card(3, "온라인 신청", "학사공지에 안내된 방법으로 신청서를 작성합니다.", "#667eea")
    response += create_step_card(4, "승인 대기", "해당 학과에서 승인 절차가 진행됩니다.", "#28a745")
    
    response += create_tip_box("신청 전 희망 전공의 교육과정을 미리 살펴보세요!")
    response += create_contact_box()
    
    return response, "APPLICATION_METHOD"


def handle_cancel(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 포기/취소 안내", "❌", "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)")
    
    response += create_info_card("포기 시기", ["매 학기 수강신청 기간 중 가능"], "#dc3545", "📆")
    response += create_info_card("포기 방법", ["학사공지 확인 후 온라인 신청"], "#fd7e14", "📋")
    response += create_info_card("유의사항", ["이수한 학점은 자유선택 학점으로 인정됩니다"], "#6c757d", "⚠️")
    
    response += create_tip_box("포기 전 학과 사무실과 상담하는 것을 권장합니다.")
    response += create_contact_box()
    
    return response, "CANCEL"


def handle_change(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 변경 안내", "🔄", "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)")
    
    response += create_info_card("종류 변경", ["복수전공 → 부전공 등: 기존 포기 후 재신청"], "#4facfe", "🔀")
    response += create_info_card("전공 변경", ["A전공 → B전공: 기존 포기 후 재신청"], "#00f2fe", "🔀")
    
    response += create_simple_card("<p style='margin:0;'>✅ 동일 학기에 <strong>포기와 신청을 동시에</strong> 처리할 수 있습니다.</p>", "#e3f2fd", "#2196f3")
    
    response += create_contact_box()
    
    return response, "CHANGE"


def handle_program_comparison(user_input, extracted_info, data_dict):
    programs_to_compare = extracted_info.get('programs', [])
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    if len(programs_to_compare) < 2:
        programs_to_compare = list(programs.keys())[:4]
    
    comparison_data = []
    for pn in programs_to_compare:
        if pn in programs:
            comparison_data.append({'name': pn, **programs[pn]})
        elif pn == '마이크로디그리' and '소단위전공과정' in programs:
            comparison_data.append({'name': '소단위전공과정', **programs['소단위전공과정']})
    
    response = create_header_card("다전공 제도 비교", "📊", "linear-gradient(135deg, #5f72bd 0%, #9b23ea 100%)")
    
    if len(comparison_data) >= 2:
        headers = ["제도"] + [d['name'] for d in comparison_data]
        rows = [
            ["이수학점"] + [d.get('credits_multi', '-') for d in comparison_data],
            ["본전공"] + [d.get('credits_primary', '-') for d in comparison_data],
            ["학위표기"] + [str(d.get('degree', '-'))[:12] for d in comparison_data],
            ["난이도"] + [str(d.get('difficulty', '-')) for d in comparison_data],
        ]
        response += create_table_html(headers, rows)
    else:
        headers = ["구분", "복수전공", "부전공", "융합전공", "마이크로디그리"]
        rows = [
            ["이수학점", "36학점", "21학점", "36학점", "12학점"],
            ["학위표기", "2개 학위", "부전공 표기", "융합전공명", "이수증"],
            ["난이도", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐"],
        ]
        response += create_table_html(headers, rows)
    
    response += create_tip_box("학점 부담이 걱정되면 부전공이나 마이크로디그리로 시작해보세요!")
    response += create_contact_box()
    
    return response, "PROGRAM_COMPARISON"


def handle_credit_info(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 제도별 이수 학점", "📖", "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)")
    
    response += create_warning_box("전공필수/전공선택 학점은 본전공과 학번에 따라 다를 수 있습니다.")
    
    headers = ["제도", "다전공 이수학점", "본전공 감축"]
    rows = [
        ["복수전공", "36학점 이상", "있음"],
        ["부전공", "21학점 이상", "있음"],
        ["융합전공", "36학점 이상", "있음"],
        ["연계전공", "36학점 이상", "있음"],
        ["마이크로디그리", "12~18학점", "없음"],
    ]
    response += create_table_html(headers, rows)
    
    response += create_tip_box("왼쪽 '다전공 제도 안내' 메뉴에서 본인 학번/전공에 맞는 상세 학점을 확인하세요!")
    response += create_contact_box()
    
    return response, "CREDIT_INFO"
    
    if len(programs_to_compare) < 2:
        programs_to_compare = list(programs.keys())[:4]
    
    comparison_data = []
    for pn in programs_to_compare:
        if pn in programs:
            comparison_data.append({'name': pn, **programs[pn]})
        elif pn == '마이크로디그리' and '소단위전공과정' in programs:
            comparison_data.append({'name': '소단위전공과정', **programs['소단위전공과정']})
    
    response = "## 📊 다전공 제도 비교\n\n"
    if len(comparison_data) >= 2:
        response += "| 구분 | " + " | ".join([d['name'] for d in comparison_data]) + " |\n"
        response += "|------" + "|------" * len(comparison_data) + "|\n"
        response += "| **이수학점** | " + " | ".join([d.get('credits_multi', '-') for d in comparison_data]) + " |\n"
        response += "| **본전공** | " + " | ".join([d.get('credits_primary', '-') for d in comparison_data]) + " |\n"
        response += "| **학위표기** | " + " | ".join([str(d.get('degree', '-'))[:15] for d in comparison_data]) + " |\n"
        response += "| **난이도** | " + " | ".join([str(d.get('difficulty', '-')) for d in comparison_data]) + " |\n"
    else:
        response += "| 구분 | 복수전공 | 부전공 | 융합전공 | 마이크로디그리 |\n"
        response += "|------|----------|--------|----------|----------------|\n"
        response += "| **이수학점** | 36학점 | 21학점 | 36학점 | 12학점 |\n"
        response += "| **학위표기** | 2개 학위 | 부전공 표기 | 융합전공명 | 이수증 |\n"
        response += "| **난이도** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |\n"
    
    response += f"\n---\n{CONTACT_MESSAGE}"
    return response, "PROGRAM_COMPARISON"


def handle_credit_info(user_input, extracted_info, data_dict):
    response = "## 📖 다전공 제도별 이수 학점\n\n"
    response += "⚠️ **전공필수/전공선택 학점은 본전공과 학번에 따라 다를 수 있습니다.**\n\n"
    response += "### 📌 기본 이수 학점\n\n"
    response += "| 제도 | 다전공 이수학점 | 본전공 감축 |\n"
    response += "|------|----------------|------------|\n"
    response += "| **복수전공** | 36학점 이상 | 있음 |\n"
    response += "| **부전공** | 21학점 이상 | 있음 |\n"
    response += "| **융합전공** | 36학점 이상 | 있음 |\n"
    response += "| **연계전공** | 36학점 이상 | 있음 |\n"
    response += "| **마이크로디그리** | 12~18학점 | 없음 |\n\n"
    response += f"---\n💡 왼쪽 '다전공 제도 안내'에서 상세 학점을 확인하세요.\n\n{CONTACT_MESSAGE}"
    return response, "CREDIT_INFO"


# 🔧 수정 #9: "다전공이 뭐야" 질문 처리 개선
def handle_program_info(user_input, extracted_info, data_dict):
    program_name = extracted_info.get('program', '')
    programs = data_dict.get('programs', PROGRAM_INFO)
    user_clean = user_input.replace(' ', '').lower()
    
    # "다전공이 뭐야?" - 전체 다전공 제도 안내
    is_general = (
        program_name == '다전공' or 
        '다전공이뭐' in user_clean or 
        '다전공뭐야' in user_clean or
        '다전공제도' in user_clean or
        (('다전공' in user_clean or '유연학사' in user_clean) and 
         any(kw in user_clean for kw in ['뭐', '무엇', '알려', '설명']))
    )
    
    if is_general:
        response = create_header_card("다전공(유연학사제도) 안내", "🎓", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
        
        response += create_simple_card("<p style='margin:0; font-size: 0.95rem;'><strong>다전공 제도</strong>는 주전공(제1전공) 외에 다른 전공을 추가로 이수할 수 있는 <strong>유연학사제도</strong>입니다.</p>", "#f0f4ff", "#667eea")
        
        # 제도 테이블
        headers = ["제도", "이수학점", "학위표기"]
        rows = []
        for p_name, p_info in programs.items():
            rows.append([
                p_name,
                p_info.get('credits_multi', '-'),
                str(p_info.get('degree', '-'))[:15]
            ])
        response += create_table_html(headers, rows)
        
        # 장점 카드
        response += create_info_card("다전공의 장점", [
            "📚 다양한 분야의 전문성 확보",
            "💼 취업 경쟁력 강화", 
            "🎓 학위기에 추가 전공 표기"
        ], "#28a745", "✨")
        
        response += create_contact_box()
        return response, "PROGRAM_INFO"
    
    # 특정 제도 설명
    program_mapping = {'복수전공': '복수전공', '부전공': '부전공', '융합전공': '융합전공',
                       '융합부전공': '융합부전공', '연계전공': '연계전공', '마이크로디그리': '소단위전공과정'}
    actual_name = program_mapping.get(program_name, program_name)
    
    if actual_name not in programs:
        for key in programs.keys():
            if program_name in key or key in program_name:
                actual_name = key
                break
    
    if actual_name not in programs:
        return f"'{program_name}' 제도 정보를 찾을 수 없습니다.\n{CONTACT_MESSAGE}", "ERROR"
    
    info = programs[actual_name]
    display_name = '소단위전공과정(마이크로디그리)' if actual_name == '소단위전공과정' else actual_name
    
    # 제도별 그라데이션 색상
    gradients = {
        '복수전공': "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        '부전공': "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
        '융합전공': "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
        '융합부전공': "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
        '연계전공': "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        '소단위전공과정': "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
    }
    gradient = gradients.get(actual_name, "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
    
    response = create_header_card(display_name, "🎓", gradient)
    
    # 개요
    response += create_simple_card(f"<p style='margin:0; font-size: 0.95rem;'>{info.get('description', '-')}</p>", "#f8f9fa", "#6c757d")
    
    # 이수학점 테이블
    headers = ["구분", "학점"]
    rows = [
        ["교양", info.get('credits_general', '-')],
        ["원전공(본전공)", info.get('credits_primary', '-')],
        ["다전공", info.get('credits_multi', '-')],
    ]
    response += create_table_html(headers, rows, ["#007bff", "#28a745", "#ffc107"])
    
    # 신청자격, 학위표기, 난이도
    response += create_info_card("신청자격", [info.get('qualification', '-')], "#007bff", "✅")
    response += create_info_card("학위표기", [info.get('degree', '-')], "#6f42c1", "📜")
    response += create_simple_card(f"<p style='margin:0;'><strong>⭐ 난이도:</strong> {info.get('difficulty', '-')}</p>", "#fff9e6", "#ffc107")
    
    response += create_contact_box()
    return response, "PROGRAM_INFO"


# 🔧 수정 #1: AI챗봇 과목 안내 - 학년/학기/이수구분별 정리
def handle_course_search(user_input, extracted_info, data_dict):
    major = extracted_info.get('major')
    courses_data = data_dict.get('courses', COURSES_DATA)
    
    if not major and not courses_data.empty:
        user_clean = user_input.replace(' ', '')
        for m in courses_data['전공명'].unique():
            m_clean = str(m).replace(' ', '')
            if m_clean in user_clean or user_clean in m_clean:
                major = m
                break
            if len(m_clean) > 3:
                keyword = m_clean.replace('전공', '').replace('융합', '')[:4]
                if keyword in user_clean:
                    major = m
                    break
    
    if not major:
        response = create_header_card("과목 조회", "📚", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
        response += create_simple_card("<p style='margin:0;'>어떤 전공의 과목을 찾으시나요?</p>", "#f0f4ff", "#667eea")
        
        # 계열별 전공 목록 표시
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
        else:
            # 계열 정보가 없으면 기존 방식
            available_majors = []
            if not courses_data.empty:
                available_majors = sorted(courses_data['전공명'].unique().tolist())[:10]
            if available_majors:
                majors_html = " ".join([f'<span style="background: #e3f2fd; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; margin: 2px; display: inline-block;">{m}</span>' for m in available_majors])
                response += f"<div style='margin: 10px 0;'><strong>📋 조회 가능한 전공:</strong><br>{majors_html}</div>"
        
        response += create_tip_box("예시: \"AI반도체융합전공 과목 알려줘\"")
        response += create_contact_box()
        return response, "COURSE_SEARCH"
    
    if courses_data.empty:
        return f"'{major}' 과목 정보를 찾을 수 없습니다.\n\n💡 **정확한 전공명을 다시 입력해주세요.**\n\n{CONTACT_MESSAGE}", "ERROR"
    
    major_courses = courses_data[courses_data['전공명'] == major]
    if major_courses.empty:
        major_keyword = major.replace('전공', '').replace('융합', '')
        major_courses = courses_data[courses_data['전공명'].str.contains(major_keyword, case=False, na=False)]
    
    if major_courses.empty:
        # 비슷한 전공 찾기 + 계열별 안내
        response = create_header_card(f"'{major}' 과목 조회 실패", "📚", "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)")
        response += create_warning_box(f"입력하신 <strong>'{major}'</strong> 전공을 찾을 수 없습니다.")
        
        major_keyword = major.replace('전공', '').replace('융합', '').replace('학과', '')[:3]
        similar_majors = []
        if major_keyword and not courses_data.empty:
            for m in courses_data['전공명'].unique():
                m_clean = str(m).replace('전공', '').replace('융합', '')
                if major_keyword in m_clean:
                    similar_majors.append(m)
        
        if similar_majors:
            similar_html = " ".join([f'<span style="background: #d4edda; color: #155724; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; margin: 2px; display: inline-block;">{m}</span>' for m in similar_majors[:5]])
            response += f"""
<div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 10px 0;">
    <strong>🔍 혹시 이 전공을 찾으셨나요?</strong><br>
    <div style="margin-top: 8px;">{similar_html}</div>
</div>
"""
        else:
            # 계열별 전공 목록 표시
            category_majors = get_majors_by_category()
            if category_majors and len(category_majors) > 1:
                response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
                response += format_majors_by_category_html(category_majors)
        
        response += create_tip_box("정확한 전공명을 다시 입력해주세요.")
        response += create_contact_box()
        return response, "COURSE_SEARCH"
    
    actual_major = major_courses['전공명'].iloc[0]
    program_types = major_courses['제도유형'].unique().tolist()
    
    response = f"## 📚 {actual_major} 교과목 안내\n\n"
    response += f"📋 **제도유형**: {', '.join([str(pt) for pt in program_types if pd.notna(pt)])}\n\n"
    
    years = sorted([int(y) for y in major_courses['학년'].dropna().unique()])
    
    for y in years:
        year_data = major_courses[major_courses['학년'] == y]
        response += f"### 📅 {y}학년\n\n"
        
        semesters = sorted([int(s) for s in year_data['학기'].dropna().unique()])
        
        for sem in semesters:
            sem_data = year_data[year_data['학기'] == sem]
            response += f"#### {sem}학기\n\n"
            
            required = sem_data[sem_data['이수구분'].str.contains('필수', na=False)]
            elective = sem_data[sem_data['이수구분'].str.contains('선택', na=False)]
            
            if not required.empty:
                response += "🔴 **전공필수**\n"
                for _, row in required.iterrows():
                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                    response += f"- {row.get('과목명', '-')} ({credit})\n"
                response += "\n"
            
            if not elective.empty:
                response += "🟢 **전공선택**\n"
                for _, row in elective.iterrows():
                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                    response += f"- {row.get('과목명', '-')} ({credit})\n"
                response += "\n"
        
        response += "---\n\n"
    
    response += f"📌 **더 자세한 교육과정은 학교 홈페이지 [학사공지]({ACADEMIC_NOTICE_URL})를 참고하세요.**\n\n"
    response += CONTACT_MESSAGE
    return response, "COURSE_SEARCH"


def handle_contact_search(user_input, extracted_info, data_dict):
    major = extracted_info.get('major')
    majors_info = data_dict.get('majors', MAJORS_INFO)
    
    if majors_info.empty:
        response = create_header_card("연락처 조회", "📞", "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)")
        response += create_warning_box("전공 정보를 불러올 수 없습니다.")
        response += create_contact_box()
        return response, "ERROR"
    
    if not major:
        user_clean = user_input.replace(' ', '')
        for _, row in majors_info.iterrows():
            m_name = str(row['전공명'])
            if m_name.replace(' ', '') in user_clean:
                major = m_name
                break
    
    if not major:
        response = create_header_card("연락처 조회", "📞", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
        response += create_simple_card("<p style='margin:0;'>어떤 전공의 연락처를 찾으시나요?</p>", "#f0f4ff", "#667eea")
        
        # 계열별 전공 목록 표시
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
        
        response += create_tip_box("예시: \"경영학전공 연락처 알려줘\"")
        response += create_contact_box()
        return response, "CONTACT_SEARCH"
    
    result = majors_info[majors_info['전공명'].str.contains(major.replace('전공', ''), case=False, na=False)]
    
    if result.empty:
        response = create_header_card("연락처 조회", "📞", "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)")
        response += create_warning_box(f"'{major}' 연락처를 찾을 수 없습니다.")
        response += create_contact_box()
        return response, "ERROR"
    
    row = result.iloc[0]
    response = create_header_card(f"{row['전공명']} 연락처", "📞", "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)")
    
    response += create_info_card("전공명", [row['전공명']], "#11998e", "🎓")
    response += create_info_card("연락처", [row.get('연락처', '-')], "#007bff", "📱")
    response += create_info_card("위치", [row.get('위치', row.get('사무실위치', '-'))], "#6f42c1", "📍")
    
    return response, "CONTACT_SEARCH"


def handle_recommendation(user_input, extracted_info, data_dict):
    response = create_header_card("맞춤형 다전공 추천", "🎯", "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)")
    
    response += create_simple_card("<p style='margin:0; font-size: 0.95rem;'>정확한 추천을 위해 아래 정보가 필요합니다</p>", "#fef0f5", "#f5576c")
    
    response += create_info_card("필요한 정보", [
        "📅 기준학번 (예: 2022학번)",
        "🎓 현재 본전공 (예: 경영학전공)",
        "📊 이수한 전공필수/전공선택 학점"
    ], "#f093fb", "📋")
    
    response += create_tip_box("예시: \"저는 2022학번 경영학전공이고, 전필 3학점, 전선 9학점 들었어요. 다전공 추천해주세요!\"")
    response += create_contact_box()
    
    return response, "RECOMMENDATION"


def handle_greeting(user_input, extracted_info, data_dict):
    response = create_header_card("안녕하세요!", "👋", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
    
    response += create_simple_card("<p style='margin:0; font-size: 1rem;'><strong>한경국립대학교 다전공(유연학사제도) 안내 AI챗봇</strong>입니다 😊</p>", "#f0f4ff", "#667eea")
    
    response += """
<div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
    <h4 style="margin: 0 0 12px 0; color: #333;">🎯 무엇을 도와드릴까요?</h4>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
        <div style="background: #e3f2fd; padding: 10px; border-radius: 8px;">
            <strong style="color: #1565c0;">📝 신청</strong><br>
            <span style="font-size: 0.85rem; color: #666;">"신청 자격이 뭐야?"</span>
        </div>
        <div style="background: #e8f5e9; padding: 10px; border-radius: 8px;">
            <strong style="color: #2e7d32;">📊 비교</strong><br>
            <span style="font-size: 0.85rem; color: #666;">"복수전공 vs 부전공"</span>
        </div>
        <div style="background: #fff3e0; padding: 10px; border-radius: 8px;">
            <strong style="color: #ef6c00;">📖 학점</strong><br>
            <span style="font-size: 0.85rem; color: #666;">"몇 학점 필요해?"</span>
        </div>
        <div style="background: #fce4ec; padding: 10px; border-radius: 8px;">
            <strong style="color: #c2185b;">🎯 추천</strong><br>
            <span style="font-size: 0.85rem; color: #666;">"다전공 추천해줘"</span>
        </div>
    </div>
</div>
"""
    
    response += create_tip_box("위의 <strong>'💡 어떤 질문을 해야 할지 모르겠나요?'</strong>를 클릭해보세요!")
    
    return response, "GREETING"


def handle_blocked(user_input, extracted_info, data_dict):
    response = create_header_card("잠깐만요!", "⚠️", "linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)")
    response += create_warning_box("부적절한 표현이 감지되었어요.")
    response += create_simple_card("<p style='margin:0;'>다전공 관련 질문을 해주시면 친절하게 답변드릴게요! 😊</p>", "#f0f7ff", "#007bff")
    return response, "BLOCKED"


def handle_out_of_scope(user_input, extracted_info, data_dict):
    response = create_header_card("모릅니다", "🚫", "linear-gradient(135deg, #636e72 0%, #b2bec3 100%)")
    
    response += create_simple_card("<p style='margin:0;'>저는 <strong>한경국립대학교 다전공(유연학사제도) 전용 AI챗봇</strong>이에요.</p>", "#f8f9fa", "#6c757d")
    
    response += """
<div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
    <h4 style="margin: 0 0 12px 0; color: #333;">💬 이런 질문은 답변할 수 있어요!</h4>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.9rem;">
        <div style="padding: 8px; background: #e3f2fd; border-radius: 6px;">📝 신청 자격/기간/방법</div>
        <div style="padding: 8px; background: #e8f5e9; border-radius: 6px;">📊 제도 비교</div>
        <div style="padding: 8px; background: #fff3e0; border-radius: 6px;">📖 이수학점 정보</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📞 전공별 연락처</div>
    </div>
</div>
"""
    
    response += create_tip_box("위의 <strong>'💡 어떤 질문을 해야 할지 모르겠나요?'</strong>를 클릭해보세요!")
    
    return response, "OUT_OF_SCOPE"


def handle_general(user_input, extracted_info, data_dict):
    return f"죄송합니다. 답변을 생성하지 못했습니다.\n{CONTACT_MESSAGE}", "ERROR"


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
    intent, method, extracted_info = classify_intent(user_input)
    handler = INTENT_HANDLERS.get(intent, handle_general)
    response, response_type = handler(user_input, extracted_info, data_dict)
    return response, response_type
# ============================================================
# 📊 이수체계도 및 과목 표시 함수
# ============================================================

# 🔧 수정 #2: 소단위전공 이미지 2개 표시 문제 해결
def display_curriculum_image(major, program_type):
    """이수체계도/과정 안내 이미지 표시 - 여러 이미지 지원"""
    is_fusion = program_type == "융합전공"
    is_micro = "소단위" in program_type or "마이크로" in program_type
    
    if not is_fusion and not is_micro:
        return
    
    if CURRICULUM_MAPPING.empty:
        return
    
    def match_program_type_for_image(type_value):
        type_str = str(type_value).strip().lower()
        if is_fusion:
            return "융합전공" in type_str and "융합부전공" not in type_str
        if is_micro:
            return any(kw in type_str for kw in ['소단위', '마이크로', 'md'])
        return False
    
    clean_major = major
    if '(' in major:
        clean_major = major.split('(')[0].strip()
    
    search_keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').strip()
    
    type_matched = CURRICULUM_MAPPING[CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image)]
    
    if type_matched.empty:
        return
    
    # 전공명 매칭
    filtered = type_matched[type_matched['전공명'] == clean_major]
    
    if filtered.empty:
        filtered = type_matched[type_matched['전공명'] == major]
    
    if filtered.empty and len(search_keyword) >= 2:
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            cm_keyword = cm_major.replace('전공', '').replace('과정', '').replace('MD', '').strip()
            if search_keyword[:3] in cm_keyword or cm_keyword[:3] in search_keyword:
                # 🔧 수정: 해당 전공의 모든 이미지 가져오기
                filtered = type_matched[type_matched['전공명'] == cm_major]
                break
    
    # 🔧 수정 #2: 모든 이미지 표시 (여러 개 지원)
    if not filtered.empty:
        images_shown = 0
        total_images = len(filtered)
        
        for _, row in filtered.iterrows():
            filename = row['파일명']
            if pd.notna(filename) and str(filename).strip():
                image_path = f"{CURRICULUM_IMAGES_PATH}/{filename}"
                if os.path.exists(image_path):
                    if is_fusion:
                        caption = f"{clean_major} 이수체계도"
                    else:
                        if total_images > 1:
                            caption = f"{clean_major} 과정 안내 ({images_shown + 1}/{total_images})"
                        else:
                            caption = f"{clean_major} 과정 안내"
                    st.image(image_path, caption=caption)
                    images_shown += 1
        
        if images_shown == 0:
            st.caption("📷 이미지 파일 준비 중입니다.")


# 🔧 수정 #3: 소단위전공 교과목 'XX MD' 패턴으로 검색
def display_courses(major, program_type):
    """과목 정보 표시 - 학년별/학기별/이수구분별 정리"""
    if COURSES_DATA.empty:
        st.info("교과목 데이터가 없습니다.")
        return False
    
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
    
    clean_major = major
    display_major = major
    
    if '(' in major:
        clean_major = major.split('(')[0].strip()
        display_major = clean_major
    
    # 1. 정확한 매칭
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == clean_major) & 
        (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
    ]
    
    # 🔧 수정 #3: 소단위전공 "XX MD" 패턴으로 검색
    if courses.empty and is_micro:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').strip()
        type_matched = COURSES_DATA[COURSES_DATA['제도유형'].apply(match_program_type_for_courses)]
        
        for course_major in type_matched['전공명'].unique():
            cm_str = str(course_major)
            if 'MD' in cm_str or 'md' in cm_str.lower():
                cm_keyword = cm_str.replace('MD', '').replace('md', '').strip()
                if len(keyword) >= 3 and len(cm_keyword) >= 3:
                    if keyword[:3] in cm_keyword or cm_keyword[:3] in keyword:
                        courses = type_matched[type_matched['전공명'] == course_major]
                        display_major = cm_str
                        break
    
    # 부분 매칭
    if courses.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '')[:4]
        if keyword:
            courses = COURSES_DATA[
                (COURSES_DATA['전공명'].str.contains(keyword, na=False)) & 
                (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
            ]
            if not courses.empty:
                display_major = courses['전공명'].iloc[0]
    
    display_program_type = "소단위전공과정(마이크로디그리)" if is_micro else program_type
    
    if not courses.empty:
        st.subheader(f"📚 ({display_program_type}) {display_major} 교과목 안내")
        
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
                        
                        required = semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)]
                        elective = semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if not required.empty:
                                st.markdown("**🔴 전공필수**")
                                for _, row in required.iterrows():
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    st.write(f"• {row.get('과목명', '')} ({credit})")
                        
                        with col2:
                            if not elective.empty:
                                st.markdown("**🟢 전공선택**")
                                for _, row in elective.iterrows():
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    st.write(f"• {row.get('과목명', '')} ({credit})")
                        
                        st.divider()
        
        # 🔧 수정 #4: 전공 연락처 표시
        st.markdown("---")
        display_major_contact(display_major)
        return True
    else:
        st.info(f"'{display_major}' 교과목 정보가 없습니다.")
        return False


# 🔧 수정 #4: 전공 문의처에 전공명, 위치 추가
def display_major_contact(major):
    """전공 연락처 표시 - 전공명, 연락처, 위치 포함"""
    if MAJORS_INFO.empty:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")
        return
    
    clean_major = major
    if '(' in major:
        clean_major = major.split('(')[0].strip()
    clean_major = clean_major.replace(' MD', '').replace('MD', '').strip()
    
    contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == clean_major]
    
    if contact_row.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '')[:4]
        if keyword:
            contact_row = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains(keyword, na=False)]
    
    if not contact_row.empty:
        row = contact_row.iloc[0]
        major_name = row.get('전공명', major)
        phone = row.get('연락처', '')
        location = row.get('사무실위치', row.get('위치', ''))
        
        contact_parts = [f"🎓 **전공명**: {major_name}"]
        if pd.notna(phone) and str(phone).strip():
            contact_parts.append(f"📞 **연락처**: {phone}")
        if pd.notna(location) and str(location).strip():
            contact_parts.append(f"📍 **사무실 위치**: {location}")
        
        st.info("**📋 전공 문의처**\n\n" + "\n\n".join(contact_parts))
    else:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")


# ============================================================
# 🖥️ 메인 UI
# ============================================================

def main():
    initialize_session_state()
    
    # 사이드바 - Modern Design
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <div style='font-size: 3rem;'>🎓</div>
            <h2 style='margin-top: 10px; font-weight: 700; color: #1F2937;'>HKNU<br>MajorBot</h2>
        </div>
        """, unsafe_allow_html=True)
        
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내", "FAQ"], 
            icons=["chat-text", "book", "question-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#6B7280", "font-size": "16px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "border-radius":"10px", "color":"#4B5563"},
                "nav-link-selected": {"background-color": "#4F46E5", "color": "white", "font-weight":"600"},
            }
        )
        
        st.markdown("---")
        
        # 팁 박스
        st.markdown("""
        <div style="background: #EEF2FF; border: 1px solid #C7D2FE; padding: 12px; border-radius: 12px; margin-bottom: 12px;">
            <p style="margin: 0; color: #4338CA; font-size: 0.85rem;">
                💡 <strong>Tip</strong><br>
                <span style="font-size: 0.8rem; color: #6366F1;">왼쪽 메뉴에서 제도를 상세히 살펴볼 수 있어요.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 참고용 안내 문구
        st.markdown("""
        <p style="color: #9CA3AF; font-size: 0.7rem; text-align: center; margin: 12px 0;">
            ⚠️ 이 AI챗봇은 단순 참고용입니다.<br>
            정확한 정보는 학사공지를 확인하세요.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Powered by 정보
        st.markdown("""
        <div style="text-align: center; padding: 8px 0;">
            <p style="color: #9CA3AF; font-size: 0.75rem; margin: 0;">
                ⚡ Powered by <strong style="color: #4F46E5;">Gemini 2.0</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if SEMANTIC_ROUTER is not None:
            st.markdown("""
            <p style="color: #9CA3AF; font-size: 0.7rem; text-align: center; margin: 4px 0;">
                🧠 Semantic Router 활성화
            </p>
            """, unsafe_allow_html=True)
    
    # 메인 콘텐츠
    if menu == "AI챗봇 상담":
        # Modern 헤더
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #111827; font-weight: 800; letter-spacing: -1px; font-size: 1.8rem;">무엇을 도와드릴까요?</h1>
            <p style="color: #6B7280; font-size: 1rem;">다전공, 복수전공, 마이크로디그리 등 궁금한 점을 물어보세요.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("✨ 자주 묻는 질문 보기", expanded=True):
            
            def click_question(q):
                st.session_state.chat_history.append({"role": "user", "content": q})
                response_text, res_type = generate_ai_response(q, st.session_state.chat_history[:-1], ALL_DATA)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
                st.rerun()
            
            # 📋 신청 관련
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 12px 0 8px 0;">
                <span style="background: #4F46E5; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">📋 신청</span>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(6)
            q_apply = [
                ("자격", "신청 자격이 뭐야?"),
                ("기간", "신청 기간 언제야?"),
                ("방법", "신청 방법 알려줘"),
                ("포기", "다전공 포기 방법"),
                ("변경", "전공 변경하고 싶어"),
                ("절차", "신청 절차 알려줘"),
            ]
            for i, (label, q) in enumerate(q_apply):
                if cols[i].button(label, key=f"qa_{i}", use_container_width=True):
                    click_question(q)
            
            # 📚 제도 관련
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
                <span style="background: #10B981; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">📚 제도</span>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(6)
            q_program = [
                ("다전공", "다전공이 뭐야?"),
                ("복수전공", "복수전공 설명해줘"),
                ("부전공", "부전공이 뭐야?"),
                ("융합전공", "융합전공 알려줘"),
                ("마이크로", "마이크로디그리 뭐야?"),
                ("비교", "복수전공 부전공 차이"),
            ]
            for i, (label, q) in enumerate(q_program):
                if cols[i].button(label, key=f"qp_{i}", use_container_width=True):
                    click_question(q)
            
            # 🎓 학점 관련
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
                <span style="background: #F59E0B; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">🎓 학점</span>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(6)
            q_credit = [
                ("이수학점", "이수 학점 알려줘"),
                ("본전공", "본전공 학점 변화"),
                ("복전학점", "복수전공 몇 학점?"),
                ("부전학점", "부전공 몇 학점?"),
                ("졸업요건", "졸업 요건 알려줘"),
                ("비교", "제도별 학점 비교"),
            ]
            for i, (label, q) in enumerate(q_credit):
                if cols[i].button(label, key=f"qc_{i}", use_container_width=True):
                    click_question(q)
            
            # 📞 전공/연락처 + 🎯 추천
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;">
                <span style="background: #EF4444; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">📞 전공 · 🎯 추천</span>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(6)
            q_etc = [
                ("연락처", "전공 연락처 알려줘"),
                ("위치", "사무실 위치 어디야?"),
                ("과목", "교과목 알려줘"),
                ("추천", "다전공 추천해줘"),
                ("쉬운거", "학점 부담 적은 거"),
                ("취업", "취업에 유리한 거"),
            ]
            for i, (label, q) in enumerate(q_etc):
                if cols[i].button(label, key=f"qe_{i}", use_container_width=True):
                    click_question(q)
        
        # 채팅 히스토리
        for chat in st.session_state.chat_history:
            avatar = "🧑‍🎓" if chat["role"] == "user" else "🤖"
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"], unsafe_allow_html=True)
        
        # 입력창
        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("답변을 생성하고 있습니다..."):
                    response_text, res_type = generate_ai_response(prompt, st.session_state.chat_history[:-1], ALL_DATA)
                    st.markdown(response_text, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
            scroll_to_bottom()
    
    elif menu == "다전공 제도 안내":
        st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #111827; font-weight: 700;">📚 다전공 제도 안내</h2>
            <p style="color: #6B7280;">학교의 다양한 다전공 제도를 한눈에 확인하세요.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 제도 카드 - Modern Design
        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"]
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '')[:50] + '...' if len(info.get('description', '')) > 50 else info.get('description', '-')
                    qual = info.get('qualification', '-')[:30] + '...' if len(str(info.get('qualification', '-'))) > 30 else info.get('qualification', '-')
                    color = colors[idx % len(colors)]
                    
                    html = f"""
                    <div style="background: white; border-radius: 16px; padding: 20px; min-height: 380px; margin-bottom: 16px;
                                border: 1px solid #E5E7EB; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); 
                                transition: transform 0.2s, box-shadow 0.2s;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                            <span style="background: {color}15; color: {color}; width: 32px; height: 32px; border-radius: 8px; 
                                        display: flex; align-items: center; justify-content: center; font-size: 16px;">🎓</span>
                            <h3 style="margin: 0; color: #111827; font-weight: 700; font-size: 1rem;">{program}</h3>
                        </div>
                        
                        <p style="color: #6B7280; font-size: 0.85rem; margin-bottom: 16px; line-height: 1.5;">{desc}</p>
                        
                        <div style="border-top: 1px solid #F3F4F6; padding-top: 12px;">
                            <div style="margin-bottom: 10px;">
                                <span style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">이수학점</span>
                                <p style="margin: 4px 0 0 0; color: #374151; font-size: 0.9rem;">
                                    본전공 {info.get('credits_primary', '-')} · 다전공 {info.get('credits_multi', '-')}
                                </p>
                            </div>
                            
                            <div style="margin-bottom: 10px;">
                                <span style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">신청자격</span>
                                <p style="margin: 4px 0 0 0; color: #374151; font-size: 0.85rem;">{qual}</p>
                            </div>
                            
                            <div style="margin-bottom: 10px;">
                                <span style="color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">학위표기</span>
                                <p style="margin: 4px 0 0 0; color: {color}; font-size: 0.85rem; font-weight: 500;">{str(info.get('degree', '-'))[:30]}</p>
                            </div>
                            
                            <div style="display: flex; justify-content: center; margin-top: 12px; padding-top: 12px; border-top: 1px solid #F3F4F6;">
                                <span style="color: #F59E0B; font-size: 0.9rem;">{info.get('difficulty', '⭐⭐⭐')}</span>
                            </div>
                        </div>
                    </div>"""
                    st.markdown(html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <h3 style="color: #111827; font-weight: 600; margin-bottom: 16px;">🔍 상세 정보 조회</h3>
        """, unsafe_allow_html=True)
        
        prog_keys = list(ALL_DATA['programs'].keys()) if 'programs' in ALL_DATA else []
        selected_program = st.selectbox("제도 선택", prog_keys)
        
        if selected_program:
            info = ALL_DATA['programs'][selected_program]
            
            tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징"])
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**개요**\n\n{info.get('description', '-')}")
                    st.markdown(f"**이수학점**: 교양 {info.get('credits_general', '-')} | 원전공 {info.get('credits_primary', '-')} | 다전공 {info.get('credits_multi', '-')}")
                    st.markdown(f"**졸업요건**: 인증 {info.get('graduation_certification', '-')} | 시험 {info.get('graduation_exam', '-')}")
                with col2:
                    st.success(f"**신청자격**\n\n{info.get('qualification', '-')}")
                    st.write(f"**학위표기**: {info.get('degree', '-')}")
            with tab2:
                for f in info.get('features', []):
                    st.write(f"✔️ {f}")
                if info.get('notes'):
                    st.warning(f"💡 {info['notes']}")
            
            st.divider()
            
            # 전공 목록
            available_majors = {}
            
            def match_program_type(type_value, selected_prog):
                type_str = str(type_value).strip()
                if "소단위" in selected_prog or "마이크로" in selected_prog:
                    return any(kw in type_str.lower() for kw in ['소단위', '마이크로', 'md'])
                if selected_prog == "부전공":
                    return "부전공" in type_str and "융합부전공" not in type_str
                if selected_prog == "융합전공":
                    return "융합전공" in type_str and "융합부전공" not in type_str
                return selected_prog in type_str
            
            if not COURSES_DATA.empty and '제도유형' in COURSES_DATA.columns:
                mask = COURSES_DATA['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                for major in COURSES_DATA[mask]['전공명'].unique():
                    available_majors[major] = None
            
            if not MAJORS_INFO.empty and '제도유형' in MAJORS_INFO.columns:
                mask = MAJORS_INFO['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                for _, row in MAJORS_INFO[mask].iterrows():
                    major_name = row['전공명']
                    edu_major = row.get('교육운영전공')
                    if pd.notna(edu_major) and str(edu_major).strip():
                        available_majors[major_name] = str(edu_major).strip()
                    elif major_name not in available_majors:
                        available_majors[major_name] = None
            
            if available_majors:
                target_programs = ["복수전공", "부전공", "융합전공", "융합부전공"]
                special_programs = ["융합전공", "융합부전공", "소단위전공과정", "연계전공"]
                
                # 계열별 전공 그룹화
                category_majors = get_majors_by_category(selected_program)
                
                if selected_program in target_programs:
                    # 특수 제도 (융합전공 등)는 계열 구분 없이 표시
                    if selected_program in special_programs or len(category_majors) <= 1:
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            all_majors = []
                            for majors in category_majors.values():
                                all_majors.extend(majors)
                            selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(set(all_majors)))
                        with col_m2:
                            # 본전공도 계열별 선택
                            primary_categories = get_majors_by_category("복수전공")
                            if len(primary_categories) > 1:
                                selected_primary_cat = st.selectbox("본전공 계열", ["선택 안 함"] + sorted(primary_categories.keys()))
                                if selected_primary_cat and selected_primary_cat != "선택 안 함":
                                    primary_list = primary_categories.get(selected_primary_cat, [])
                                    my_primary = st.selectbox("나의 본전공", ["선택 안 함"] + sorted(primary_list))
                                else:
                                    my_primary = "선택 안 함"
                            else:
                                primary_list = []
                                if not PRIMARY_REQ.empty:
                                    primary_list = sorted(PRIMARY_REQ['전공명'].unique().tolist())
                                my_primary = st.selectbox("나의 본전공", ["선택 안 함"] + primary_list)
                    else:
                        # 일반 제도 (복수전공, 부전공)는 계열별 선택
                        st.markdown("""
                        <div style="background: #e3f2fd; padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;">
                            <p style="margin: 0; font-size: 0.9rem; color: #1565c0;">
                                📌 <strong>계열을 먼저 선택</strong>하면 해당 계열의 전공 목록이 표시됩니다.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_cat, col_major = st.columns(2)
                        with col_cat:
                            category_list = sorted(category_majors.keys())
                            selected_category = st.selectbox(f"📚 {selected_program} 계열 선택", category_list)
                        
                        with col_major:
                            if selected_category:
                                major_list = category_majors.get(selected_category, [])
                                selected_major = st.selectbox(f"🎓 이수하려는 {selected_program}", sorted(major_list))
                            else:
                                selected_major = None
                        
                        # 본전공 선택 (계열별)
                        col_pri_cat, col_pri_major = st.columns(2)
                        with col_pri_cat:
                            primary_categories = get_majors_by_category("복수전공")
                            if len(primary_categories) > 1:
                                selected_primary_cat = st.selectbox("🏠 본전공 계열", ["선택 안 함"] + sorted(primary_categories.keys()))
                            else:
                                selected_primary_cat = "선택 안 함"
                        
                        with col_pri_major:
                            if selected_primary_cat and selected_primary_cat != "선택 안 함":
                                primary_list = primary_categories.get(selected_primary_cat, [])
                                my_primary = st.selectbox("🏠 나의 본전공", ["선택 안 함"] + sorted(primary_list))
                            else:
                                my_primary = "선택 안 함"
                else:
                    # 소단위전공과정 등
                    all_majors = []
                    for majors in category_majors.values():
                        all_majors.extend(majors)
                    selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(set(all_majors)))
                    my_primary = "선택 안 함"
                
                if selected_major:
                    if selected_program in target_programs:
                        admission_year = st.number_input("본인 학번", min_value=2018, max_value=datetime.now().year, value=datetime.now().year)
                        
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.subheader(f"🎯 {selected_program} 이수학점")
                            if not GRADUATION_REQ.empty:
                                req_data = GRADUATION_REQ[
                                    (GRADUATION_REQ['전공명'] == selected_major) & 
                                    (GRADUATION_REQ['제도유형'].str.contains(selected_program, na=False))
                                ].copy()
                                if not req_data.empty:
                                    req_data['기준학번'] = pd.to_numeric(req_data['기준학번'], errors='coerce')
                                    applicable = req_data[req_data['기준학번'] <= admission_year].sort_values('기준학번', ascending=False)
                                    if not applicable.empty:
                                        row = applicable.iloc[0]
                                        st.write(f"전공필수: **{int(row.get('다전공_전공필수', 0))}**학점")
                                        st.write(f"전공선택: **{int(row.get('다전공_전공선택', 0))}**학점")
                                        st.markdown(f"#### 👉 합계 {int(row.get('다전공_계', 0))}학점")
                        
                        with col_r:
                            st.subheader(f"🏠 본전공 학점 변화")
                            if my_primary != "선택 안 함" and not PRIMARY_REQ.empty:
                                pri_data = PRIMARY_REQ[PRIMARY_REQ['전공명'] == my_primary].copy()
                                if not pri_data.empty:
                                    pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
                                    pri_valid = pri_data[pri_data['기준학번'] <= admission_year].sort_values('기준학번', ascending=False)
                                    for _, p_row in pri_valid.iterrows():
                                        if selected_program in str(p_row['제도유형']):
                                            st.write(f"전공필수: **{int(p_row.get('본전공_전공필수', 0))}**학점")
                                            st.write(f"전공선택: **{int(p_row.get('본전공_전공선택', 0))}**학점")
                                            st.markdown(f"#### 👉 합계 {int(p_row.get('본전공_계', 0))}학점")
                                            break
                            else:
                                st.info("본전공을 선택하면 변동 학점을 확인할 수 있습니다.")
                    
                    st.divider()
                    
                    # 이수체계도 및 교과목 표시
                    if selected_program == "융합전공":
                        st.subheader("📋 이수체계도")
                        display_curriculum_image(selected_major, selected_program)
                        display_courses(selected_major, selected_program)
                    elif "소단위" in selected_program or "마이크로" in selected_program:
                        st.subheader("🖼️ 과정 안내 이미지")
                        display_curriculum_image(selected_major, selected_program)
                        display_courses(selected_major, selected_program)
                    else:
                        display_courses(selected_major, selected_program)
    
    elif menu == "FAQ":
        st.header("❓ 자주 묻는 질문")
        
        if FAQ_DATA:
            categories = list(set([faq.get('카테고리', '일반') for faq in FAQ_DATA if faq.get('카테고리')]))
            selected_cat = st.selectbox("카테고리", ["전체"] + sorted(categories))
            search = st.text_input("🔍 검색", placeholder="키워드 입력...")
            
            filtered = FAQ_DATA
            if selected_cat != "전체":
                filtered = [f for f in filtered if f.get('카테고리') == selected_cat]
            if search:
                filtered = [f for f in filtered if search.lower() in f.get('질문', '').lower() or search.lower() in f.get('답변', '').lower()]
            
            st.write(f"📋 {len(filtered)}개 FAQ")
            for faq in filtered:
                with st.expander(f"**Q. {faq.get('질문', '')}**"):
                    st.markdown(f"**A.** {faq.get('답변', '')}")
        else:
            st.warning("FAQ 데이터가 없습니다.")
        
        st.divider()
        st.info("💡 원하는 답변이 없으면 **AI챗봇 상담**에서 직접 질문하세요!")


if __name__ == "__main__":
    initialize_session_state()
    main()
