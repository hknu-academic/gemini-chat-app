"""
============================================================
🎓 한경국립대학교 다전공 안내 AI챗봇
============================================================
버전: 3.8 (신청 방법 상세화)
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
11. HTML 카드 스타일 UI 적용 (컬러박스 + 이모지)
12. 사이드바 AI챗봇/다전공 소개 스타일링
13. 질문 버튼 전체 그리드 방식 (24개 항목)
14. 계열별 전공 그룹화 (다전공 제도 안내 + AI챗봇)
15. 신청 방법 전공 유형별 상세화 (복수/부/연계/융합/소단위)
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
    from semantic_router.encoders import GoogleEncoder
    from semantic_router.index import LocalIndex
    SEMANTIC_ROUTER_AVAILABLE = True
    SEMANTIC_ROUTER_VERSION = "0.1.x"
except ImportError:
    try:
        from semantic_router import Route
        from semantic_router.layer import RouteLayer as SemanticRouter
        from semantic_router.encoders import GoogleEncoder
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
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.hknu.ac.kr', # 도움말 링크
        'Report a bug': 'https://www.hknu.ac.kr', # 버그 보고 링크
        'About': "# 한경국립대학교 다전공 안내 AI 챗봇" # About 텍스트
    }
)

# 🔧 수정 #6, #7: CSS - Streamlit 브랜딩 완전 숨김 + 모바일 가독성 개선
hide_streamlit_style = """
<style>
    /* 1. 상단 무지개 장식선만 숨김 */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* 2. 하단 푸터만 숨김 */
    footer {
        display: none !important;
    }

    /* 3. 본문 여백만 살짝 조정 */
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    /* 
       [중요] 헤더(header)와 툴바(stToolbar)를 숨기는 코드를 모두 뺐습니다.
       이렇게 하면 오른쪽 위에 '점 3개' 메뉴는 보이겠지만,
       왼쪽 위의 '사이드바 열기 버튼'은 무조건 살아있게 됩니다.
    */
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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
        "자격이 뭐야?", "자격 알려줘", "조건이 뭐야?",
    ],
    'APPLICATION_PERIOD': [
        "신청 기간이 언제예요?", "언제 신청해요?", "마감일이 언제야?",
        "지원 기간 알려주세요", "언제까지 신청할 수 있어요?", "접수 기간이 어떻게 돼?",
        "몇 월에 신청해?", "신청 시작일이 언제야?", "지금 신청 가능해?",
        "기간은 언제야?", "기간 알려줘", "언제부터 언제까지야?", "기간이 어떻게 돼?",
    ],
    'APPLICATION_METHOD': [
        "신청 방법이 어떻게 되나요?", "어떻게 신청해요?", "신청 절차 알려주세요",
        "지원하려면 어떻게 해야 해?", "신청하는 법 알려줘", "어디서 신청해?",
        "절차가 어떻게 돼?", "지원 방법이 뭐야?",
        "신청 방법은 뭐야?", "방법 알려줘", "어떻게 하는 거야?",
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
    'QUALIFICATION': ['신청자격', '지원자격', '자격요건', '자격이뭐', '누가신청', '신청조건', '자격알려', '조건이뭐'],
    'APPLICATION_PERIOD': ['신청기간', '지원기간', '접수기간', '언제신청', '마감일', '언제까지', '기간은언제', '기간알려', '언제부터'],
    'APPLICATION_METHOD': ['신청방법', '지원방법', '신청절차', '어떻게신청', '어디서신청', '방법은뭐', '방법알려', '어떻게하는'],
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
        encoder = GoogleEncoder(
            name="models/text-embedding-004",  # 구글의 최신 임베딩 모델 (한국어 성능 우수)
            api_key=st.secrets["GEMINI_API_KEY"]           # 코드 상단에 정의된 API 키 변수 사용
        )
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
    special_programs = ["융합전공", "융합부전공", "소단위전공과정"]
    
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
    """계열별 색상 반환"""
    colors = {
        '공학계열': '#e74c3c',
        '자연과학계열': '#27ae60',
        '인문사회계열': '#3498db',
        '예체능계열': '#9b59b6',
        '의학계열': '#e67e22',
        '사범계열': '#1abc9c',
        '기타': '#95a5a6',
        '전체': '#667eea',
    }
    return colors.get(category, '#6c757d')


def format_majors_by_category_html(category_majors):
    """계열별 전공 목록을 HTML 카드로 포맷팅"""
    if not category_majors:
        return "<p>전공 정보가 없습니다.</p>"
    
    html = ""
    for category, majors in category_majors.items():
        if not majors:
            continue
        color = get_category_color(category)
        majors_tags = " ".join([f'<span style="background: {color}22; color: {color}; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; margin: 2px; display: inline-block;">{m}</span>' for m in majors])
        
        html += f"""
<div style="margin-bottom: 12px;">
    <div style="background: {color}; color: white; padding: 6px 12px; border-radius: 8px 8px 0 0; font-weight: bold; font-size: 0.9rem;">
        📚 {category} ({len(majors)}개)
    </div>
    <div style="background: #f8f9fa; padding: 10px; border-radius: 0 0 8px 8px; border: 1px solid #dee2e6; border-top: none;">
        {majors_tags}
    </div>
</div>
"""
    return html


# ============================================================
# 🎨 옵션 A: 컬러박스 + 이모지 강화 스타일
# ============================================================

def create_header_card(title, emoji="📋", color="#667eea"):
    """상단 헤더 카드 생성 - 단순 텍스트"""
    return f"""
<h3 style="margin: 20px 0 16px 0; font-size: 1.3rem; color: #333; font-weight: 600;">
    {emoji} {title}
</h3>
"""

def create_info_card(title, content_list, border_color="#007bff", emoji="📌"):
    """정보 카드 생성 - 단순 텍스트"""
    items_html = ""
    for item in content_list:
        items_html += f'<p style="margin: 6px 0 6px 20px; font-size: 0.95rem; color: #333;">• {item}</p>\n'
    
    return f"""
<div style="margin: 12px 0;">
    <h4 style="color: #333; margin: 10px 0 8px 0; font-size: 1rem; font-weight: 600;">{emoji} {title}</h4>
    {items_html}
</div>
"""

def create_simple_card(content, bg_color="#f0f7ff", border_color="#007bff"):
    """간단한 정보 카드 - 단순 텍스트"""
    return f"""
<div style="margin: 12px 0; padding: 0;">
    {content}
</div>
"""

def create_step_card(step_num, title, description, color="#007bff"):
    """단계별 카드 생성"""
    return f"""
<div style="display: flex; align-items: flex-start; margin: 12px 0; padding: 12px; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
    <div style="background: {color}; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 14px; flex-shrink: 0;">{step_num}</div>
    <div>
        <strong style="color: #333; font-size: 0.95rem;">{title}</strong>
        <p style="margin: 4px 0 0 0; color: #666; font-size: 0.9rem;">{description}</p>
    </div>
</div>
"""

def create_tip_box(text, emoji="💡"):
    """팁 박스 생성 - 단순 텍스트"""
    return f"""
<p style="margin: 12px 0; color: #666; font-size: 0.9rem; font-style: italic;">
    {emoji} <strong>TIP:</strong> {text}
</p>
"""

def create_warning_box(text, emoji="⚠️"):
    """경고 박스 생성 - 단순 텍스트"""
    return f"""
<p style="margin: 12px 0; color: #dc3545; font-size: 0.9rem; font-weight: 500;">
    {emoji} {text}
</p>
"""

def create_contact_box():
    """연락처 박스 생성 - 단순 텍스트"""
    return """
<p style="margin: 16px 0 0 0; color: #666; font-size: 0.9rem;">
    📞 <strong>문의:</strong> 전공 사무실 또는 학사지원팀 <strong>031-670-5035</strong>
</p>
"""

def create_table_html(headers, rows, colors=None):
    """HTML 테이블 생성 - 단순 스타일"""
    header_html = "".join([f'<th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd; font-weight: 600;">{h}</th>' for h in headers])
    
    rows_html = ""
    for idx, row in enumerate(rows):
        cells = ""
        for i, cell in enumerate(row):
            cells += f'<td style="padding: 10px; border-bottom: 1px solid #eee;">{cell}</td>'
        rows_html += f"<tr>{cells}</tr>"
    
    # HTML을 한 줄로 반환하여 Streamlit 렌더링 문제 방지
    return f'<div style="overflow-x: auto; margin: 16px 0;"><table style="width: 100%; border-collapse: collapse;"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'

def create_program_badge(program_name, color="#007bff"):
    """프로그램 배지 생성"""
    return f'<span style="background: {color}; color: white; padding: 4px 10px; border-radius: 15px; font-size: 0.85rem; margin-right: 6px;">{program_name}</span>'


# ============================================================
# 🎯 핸들러 함수들
# ============================================================

def handle_qualification(user_input, extracted_info, data_dict):
    programs = data_dict.get('programs', PROGRAM_INFO)
    
    response = create_header_card("다전공 제도별 신청 자격 요건", "📋", "#667eea")
    
    # 공통 신청 자격
    response += """
<p style="margin: 12px 0; font-size: 0.95rem; color: #333; line-height: 1.6;">
    <strong>✅ 모든 다전공 제도는 입학 후 첫 학기부터 신청 가능합니다.</strong>
</p>
<p style="margin: 12px 0 16px 0; font-size: 0.9rem; color: #666;">
    • 복수전공, 부전공, 융합전공, 융합부전공, 연계전공, 소단위전공과정(마이크로디그리) 모두 동일한 자격 요건이 적용됩니다.
</p>
"""
    
    response += create_tip_box("학점이 부족하면 부전공이나 마이크로디그리부터 시작해보세요!")
    response += create_contact_box()
    
    return response, "QUALIFICATION"


def handle_application_period(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 신청 기간 안내", "📅", "#11998e")
    
    response += """
<p style="margin: 12px 0; font-size: 0.95rem; color: #333;">
    다전공 신청은 <strong>매 학기 2회</strong> 진행됩니다.
</p>
"""
    
    # 테이블
    headers = ["이수 희망 학기", "신청 시기"]
    rows = [
        ["1학기 이수 희망", f"{APP_PERIOD_1ST}"],
        ["2학기 이수 희망", f"{APP_PERIOD_2ND}"]
    ]
    response += create_table_html(headers, rows)
    
    # 정확한 일정과 문의는 마지막에 표시
    response += f"""
<p style="margin: 16px 0 8px 0; color: #dc3545; font-size: 0.9rem; font-weight: 500;">
    ⚠️ 정확한 일정은 <a href="{ACADEMIC_NOTICE_URL}" style="color: #dc3545; text-decoration: underline;">학사공지</a>를 반드시 확인하세요!
</p>
"""
    response += create_contact_box()
    
    return response, "APPLICATION_PERIOD"


def handle_application_method(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 신청 방법", "📝", "#f093fb")
    
    # 복수전공/부전공
    response += '<div style="margin: 20px 0 10px 0;"><h4 style="color: #667eea; margin: 0; font-size: 1.1rem; font-weight: 600;">📘 복수전공/부전공</h4></div>'
    response += create_step_card(1, "신청서 작성", "복수전공/부전공 신청서를 작성합니다.", "#667eea")
    response += create_step_card(2, "원전공 지도교수 및 학부장 확인", "소속 전공의 지도교수와 학부장 확인을 받습니다.", "#764ba2")
    response += create_step_card(3, "복수전공/부전공 희망 학부장 확인", "희망하는 전공의 학부장 확인을 받습니다.", "#667eea")
    response += create_step_card(4, "복수전공/부전공 희망전공 사무실에 제출", "모든 확인이 완료된 신청서를 희망 전공 사무실에 제출합니다.", "#764ba2")
    
    # 연계전공
    response += '<div style="margin: 25px 0 10px 0;"><h4 style="color: #f093fb; margin: 0; font-size: 1.1rem; font-weight: 600;">🔗 연계전공</h4></div>'
    response += create_step_card(1, "신청서 작성", "연계전공 신청서를 작성합니다.", "#f093fb")
    response += create_step_card(2, "원전공 지도교수 및 학부장 확인", "소속 전공의 지도교수와 학부장 확인을 받습니다.", "#f5576c")
    response += create_step_card(3, "연계전공 희망 학부장 확인", "연계전공 학부장 확인을 받습니다.", "#f093fb")
    response += create_step_card(4, "연계전공 희망전공 사무실에 제출", "모든 확인이 완료된 신청서를 연계전공 사무실에 제출합니다.", "#f5576c")
    
    # 융합전공/융합부전공
    response += '<div style="margin: 25px 0 10px 0;"><h4 style="color: #4facfe; margin: 0; font-size: 1.1rem; font-weight: 600;">🌐 융합전공/융합부전공</h4></div>'
    response += create_step_card(1, "신청서 작성", "융합전공/융합부전공 신청서를 작성합니다.", "#4facfe")
    response += create_step_card(2, "원전공 지도교수 및 학부장 확인", "소속 전공의 지도교수와 학부장 확인을 받습니다.", "#00f2fe")
    response += create_step_card(3, "융합전공 학부장 확인 및 제출", "융합전공 학부장 확인을 받고 <strong>제1공학관 222호</strong>에 제출합니다.", "#4facfe")
    
    # 소단위전공과정(마이크로디그리)
    response += '<div style="margin: 25px 0 10px 0;"><h4 style="color: #fa709a; margin: 0; font-size: 1.1rem; font-weight: 600;">🎯 소단위전공과정(마이크로디그리)</h4></div>'
    response += create_step_card(1, "신청서 작성", "소단위전공과정 신청서를 작성합니다.", "#fa709a")
    response += create_step_card(2, "교육운영전공 지도교수 및 학부장 확인", "교육운영전공의 지도교수와 학부장 확인을 받습니다.", "#fee140")
    response += create_step_card(3, "교육운영전공 학부장 확인 및 사무실 제출", "교육운영전공 학부장 확인을 받고 해당 사무실에 제출합니다.", "#fa709a")
    
    response += create_tip_box("신청 전 희망 전공의 교육과정을 미리 살펴보세요!")
    response += create_contact_box()
    
    return response, "APPLICATION_METHOD"


def handle_cancel(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 포기/취소 안내", "❌", "#ff6b6b")
    
    response += create_info_card("포기 시기", 
        ["별도의 신청 기간 없이 언제든지 가능합니다"], 
        "#dc3545", "📆")
    
    response += create_info_card("포기 방법", 
        ["해당 다전공 사무실에 포기서를 제출하면 됩니다"], 
        "#fd7e14", "📋")
    
    response += create_info_card("학점 처리", 
        ["이미 취득한 학점의 이수구분은 자유선택으로 변경됩니다",
         "이수 중인 과목은 성적 확정 후 자유선택으로 변경됩니다"], 
        "#6c757d", "⚠️")
    
    response += create_tip_box("포기 전 전공 사무실과 상담하는 것을 권장합니다.")
    response += create_contact_box()
    
    return response, "CANCEL"


def handle_change(user_input, extracted_info, data_dict):
    response = create_header_card("다전공 변경 안내", "🔄", "#4facfe")
    
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
    
    response = create_header_card("다전공 제도 비교", "📊", "#5f72bd")
    
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
    response = create_header_card("다전공 제도별 이수 학점", "📖", "#ff9a9e")
    
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
        response = create_header_card("다전공(유연학사제도) 안내", "🎓", "#667eea")
        
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
    
    # 제도별 색상
    colors = {
        '복수전공': "#667eea",
        '부전공': "#11998e",
        '융합전공': "#f093fb",
        '융합부전공': "#4facfe",
        '연계전공': "#fa709a",
        '소단위전공과정': "#a8edea",
    }
    color = colors.get(actual_name, "#667eea")
    
    response = create_header_card(display_name, "🎓", color)
    
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
        response = create_header_card("과목 조회", "📚", "#667eea")
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
        major_keyword = major.replace('전공', '').replace('융합', '').replace('(', '').replace(')', '')
        major_courses = courses_data[courses_data['전공명'].str.contains(major_keyword, case=False, na=False, regex=False)]
    
    if major_courses.empty:
        # 비슷한 전공 찾기 + 계열별 안내
        response = create_header_card(f"'{major}' 과목 조회 실패", "📚", "#ff6b6b")
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
        response = create_header_card("연락처 조회", "📞", "#ff6b6b")
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
        response = create_header_card("연락처 조회", "📞", "#667eea")
        response += create_simple_card("<p style='margin:0;'>어떤 전공의 연락처를 찾으시나요?</p>", "#f0f4ff", "#667eea")
        
        # 계열별 전공 목록 표시
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
        
        response += create_tip_box("예시: \"경영학전공 연락처 알려줘\"")
        response += create_contact_box()
        return response, "CONTACT_SEARCH"
    
    result = majors_info[majors_info['전공명'].str.contains(major.replace('전공', '').replace('(', '').replace(')', ''), case=False, na=False, regex=False)]
    
    if result.empty:
        response = create_header_card("연락처 조회", "📞", "#ff6b6b")
        response += create_warning_box(f"'{major}' 연락처를 찾을 수 없습니다.")
        response += create_contact_box()
        return response, "ERROR"
    
    row = result.iloc[0]
    response = create_header_card(f"{row['전공명']} 연락처", "📞", "#11998e")
    
    response += create_info_card("전공명", [row['전공명']], "#11998e", "🎓")
    response += create_info_card("연락처", [row.get('연락처', '-')], "#007bff", "📱")
    response += create_info_card("위치", [row.get('위치', row.get('사무실위치', '-'))], "#6f42c1", "📍")
    
    return response, "CONTACT_SEARCH"


# ============================================================
# 🆕 다전공 추천 계산 함수
# ============================================================

def calculate_specific_major_recommendation(admission_year, primary_major, completed_required, completed_elective, desired_major, data_dict):
    """
    특정 희망 전공에 대한 상세 이수 학점 계산
    
    Parameters:
    - admission_year: 입학년도
    - primary_major: 본전공 이름
    - completed_required: 이미 이수한 본전공 전공필수 학점
    - completed_elective: 이미 이수한 본전공 전공선택 학점
    - desired_major: 희망하는 다전공 이름
    - data_dict: 전체 데이터
    
    Returns:
    - 상세 추천 결과 텍스트
    """
    
    result = ""
    
    # 데이터 가져오기
    primary_req = data_dict.get('primary_req', pd.DataFrame())
    grad_req = data_dict.get('grad_req', pd.DataFrame())
    majors_info = data_dict.get('majors', pd.DataFrame())
    
    if primary_req.empty or grad_req.empty:
        return "⚠️ 데이터가 없어 계산이 불가능합니다."
    
    # 희망 전공 정보 찾기
    desired_major_info = majors_info[majors_info['전공명'] == desired_major]
    
    if desired_major_info.empty:
        return f"⚠️ '{desired_major}' 전공을 찾을 수 없습니다.<br><br>💡 정확한 전공명을 입력해주세요."
    
    # 제도 유형 확인
    program_type = desired_major_info.iloc[0]['제도유형']
    
    # 1. 본전공 변경 학점 찾기
    primary_data = primary_req[
        (primary_req['전공명'] == primary_major) & 
        (primary_req['제도유형'] == program_type)
    ].copy()
    primary_data['기준학번'] = pd.to_numeric(primary_data['기준학번'], errors='coerce')
    primary_data = primary_data[primary_data['기준학번'] <= admission_year]
    primary_data = primary_data.sort_values('기준학번', ascending=False)
    
    if primary_data.empty:
        return f"⚠️ '{primary_major}' 전공의 '{program_type}' 이수요건을 찾을 수 없습니다."
    
    primary_row = primary_data.iloc[0]
    new_primary_required = int(primary_row.get('본전공_전필', 0))
    new_primary_elective = int(primary_row.get('본전공_전선', 0))
    new_primary_total = int(primary_row.get('본전공_계', 0))
    
    # 2. 남은 본전공 학점 계산
    remaining_primary_required = max(0, new_primary_required - completed_required)
    remaining_primary_elective = max(0, new_primary_elective - completed_elective)
    remaining_primary_total = remaining_primary_required + remaining_primary_elective
    
    # 3. 다전공 이수 학점 찾기
    multi_data = grad_req[
        (grad_req['전공명'] == desired_major) & 
        (grad_req['제도유형'] == program_type)
    ].copy()
    multi_data['기준학번'] = pd.to_numeric(multi_data['기준학번'], errors='coerce')
    multi_data = multi_data[multi_data['기준학번'] <= admission_year]
    multi_data = multi_data.sort_values('기준학번', ascending=False)
    
    if multi_data.empty:
        return f"⚠️ '{desired_major}'의 졸업요건을 찾을 수 없습니다."
    
    multi_row = multi_data.iloc[0]
    multi_required = int(multi_row.get('전공필수', 0))
    multi_elective = int(multi_row.get('전공선택', 0))
    multi_total = multi_required + multi_elective
    
    # 4. 총 이수해야 할 학점
    total_remaining = remaining_primary_total + multi_total
    
    # 5. 평가
    if total_remaining <= 40:
        rating = "🟢 매우 유리"
        rating_color = "#28a745"
        comment = "학점 부담이 적어 이수하기 좋습니다!"
    elif total_remaining <= 55:
        rating = "🟡 보통"
        rating_color = "#ffc107"
        comment = "적절한 계획이 필요합니다."
    else:
        rating = "🔴 부담 큼"
        rating_color = "#dc3545"
        comment = "학점 부담이 큽니다. 신중히 고려하세요."
    
    # HTML 결과 생성
    result += f"""
    <div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
        <h4 style="margin: 0 0 12px 0; color: #f093fb;">📊 상세 이수 계획</h4>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">구분</th>
                    <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">전공필수</th>
                    <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">전공선택</th>
                    <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">합계</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;"><strong>현재 이수</strong></td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{completed_required}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{completed_elective}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{completed_required + completed_elective}학점</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;"><strong>본전공 변경</strong></td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{new_primary_required}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{new_primary_elective}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{new_primary_total}학점</td>
                </tr>
                <tr style="background: #fff3e0;">
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;"><strong>남은 본전공</strong></td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{remaining_primary_required}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{remaining_primary_elective}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;"><strong>{remaining_primary_total}학점</strong></td>
                </tr>
                <tr style="background: #e3f2fd;">
                    <td style="padding: 10px; border-bottom: 1px solid #dee2e6;"><strong>{desired_major} 이수</strong></td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{multi_required}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;">{multi_elective}학점</td>
                    <td style="padding: 10px; text-align: center; border-bottom: 1px solid #dee2e6;"><strong>{multi_total}학점</strong></td>
                </tr>
                <tr style="background: #f0f4ff;">
                    <td style="padding: 10px;"><strong>총 추가 이수</strong></td>
                    <td style="padding: 10px; text-align: center;">-</td>
                    <td style="padding: 10px; text-align: center;">-</td>
                    <td style="padding: 10px; text-align: center;"><strong style="color: {rating_color}; font-size: 1.1rem;">{total_remaining}학점 {rating}</strong></td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    
    # 이수 계획
    result += f"""
    <div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
        <h4 style="margin: 0 0 12px 0; color: #f093fb;">💡 이수 계획</h4>
        <p style="margin: 8px 0; color: #333; line-height: 1.6;">
            1️⃣ <strong>남은 본전공 이수</strong>: {remaining_primary_total}학점<br>
            <span style="margin-left: 25px; color: #666; font-size: 0.9rem;">• 전공필수: {remaining_primary_required}학점</span><br>
            <span style="margin-left: 25px; color: #666; font-size: 0.9rem;">• 전공선택: {remaining_primary_elective}학점</span>
        </p>
        <p style="margin: 12px 0; color: #333; line-height: 1.6;">
            2️⃣ <strong>{desired_major} 이수</strong>: {multi_total}학점<br>
            <span style="margin-left: 25px; color: #666; font-size: 0.9rem;">• 전공필수: {multi_required}학점</span><br>
            <span style="margin-left: 25px; color: #666; font-size: 0.9rem;">• 전공선택: {multi_elective}학점</span>
        </p>
        <p style="margin: 12px 0; padding: 12px; background: #f8f9fa; border-radius: 8px; color: #333;">
            📌 총 <strong style="color: {rating_color};">{total_remaining}학점</strong>을 추가로 이수하면 <strong>{program_type}</strong>을 완료할 수 있습니다.
        </p>
        <p style="margin: 8px 0; color: #666; font-size: 0.95rem;">
            💬 <strong>평가</strong>: {comment}
        </p>
    </div>
    """
    
    # 연락처 추가
    if not desired_major_info.empty and pd.notna(desired_major_info.iloc[0].get('연락처')):
        contact = desired_major_info.iloc[0]['연락처']
        location = desired_major_info.iloc[0].get('위치', desired_major_info.iloc[0].get('사무실위치', ''))
        
        result += f"""
        <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 12px; border-radius: 8px; margin: 12px 0;">
            <p style="margin: 0; color: #333; font-size: 0.9rem;">
                📞 <strong>문의</strong>: {desired_major}<br>
                <span style="margin-left: 25px; color: #666;">• 연락처: {contact}</span>
        """
        if location:
            result += f"""<br><span style="margin-left: 25px; color: #666;">• 위치: {location}</span>"""
        result += """
            </p>
        </div>
        """
    
    result += """
    <div style="background: #f0f7ff; padding: 10px; border-radius: 8px; margin: 12px 0;">
        <p style="margin: 0; color: #666; font-size: 0.85rem;">
            💡 <strong>참고</strong>: 위 계산은 학점 기준이며, 실제 이수 과목은 각 전공의 교육과정을 확인하세요.
        </p>
    </div>
    """
    
    return result


def handle_recommendation(user_input, extracted_info, data_dict):
    import re
    
    # 질문에서 학번, 전공, 학점 정보 추출
    year_match = re.search(r'(\d{4})학번', user_input)
    major_match = re.search(r'([가-힣]+전공)', user_input)
    required_match = re.search(r'전필\s*(\d+)학점', user_input)
    elective_match = re.search(r'전선\s*(\d+)학점', user_input)
    
    # 정보가 모두 있는지 확인
    if not (year_match and major_match and (required_match or elective_match)):
        response = create_header_card("맞춤형 다전공 추천", "🎯", "#f093fb")
        response += create_simple_card("<p style='margin:0; font-size: 0.95rem;'>정확한 추천을 위해 아래 정보가 필요합니다</p>", "#fef0f5", "#f5576c")
        response += create_info_card("필요한 정보", [
            "📅 기준학번 (예: 2022학번)",
            "🎓 현재 본전공 (예: 경영학전공)",
            "📊 이수한 전공필수/전공선택 학점"
        ], "#f093fb", "📋")
        response += create_tip_box("예시: \"저는 2022학번 경영학전공이고, 전필 3학점, 전선 9학점 들었어요. 다전공 추천해주세요!\"")
        response += create_contact_box()
        return response, "RECOMMENDATION"
    
    # 정보 추출
    admission_year = int(year_match.group(1))
    primary_major = major_match.group(1)
    completed_required = int(required_match.group(1)) if required_match else 0
    completed_elective = int(elective_match.group(1)) if elective_match else 0
    total_credits = completed_required + completed_elective
    
    # 추천 시작
    response = create_header_card("맞춤형 다전공 추천", "🎯", "#f093fb")
    
    # 입력 정보 표시
    response += create_info_card("입력하신 정보", [
        f"📅 학번: {admission_year}학번",
        f"🎓 본전공: {primary_major}",
        f"📊 이수 학점: 전필 {completed_required}학점, 전선 {completed_elective}학점 (총 {total_credits}학점)"
    ], "#667eea", "📋")
    
    # MAJORS_INFO에서 추천 가능한 전공 찾기
    majors_info = data_dict.get('majors', pd.DataFrame())
    
    if majors_info.empty:
        response += create_simple_card("<p style='margin:0;'>현재 데이터에서 추천 가능한 전공을 찾을 수 없습니다. 학사지원팀에 문의해주세요.</p>", "#fff3e0", "#ff9800")
        response += create_contact_box()
        return response, "RECOMMENDATION"
    
    # 학점 기준으로 추천 전공 선택
    if total_credits < 12:
        # 부전공 추천
        recommended_majors = majors_info[
            majors_info['제도유형'].str.contains('부전공', na=False) & 
            ~majors_info['제도유형'].str.contains('융합부전공', na=False)
        ]['전공명'].head(3).tolist()
        recommendation_reason = f"현재 {total_credits}학점으로 부전공(21학점)이 적합합니다"
    else:
        # 복수전공 추천
        recommended_majors = majors_info[
            majors_info['제도유형'].str.contains('복수전공', na=False)
        ]['전공명'].head(3).tolist()
        recommendation_reason = f"현재 {total_credits}학점으로 복수전공(36학점) 도전 가능합니다"
    
    if recommended_majors:
        response += '<div style="margin: 20px 0;"><h4 style="color: #f093fb; margin: 0; font-size: 1.1rem; font-weight: 600;">💡 추천 다전공 상세 분석</h4></div>'
        response += f'<p style="margin: 10px 0; color: #666; font-size: 0.9rem;">{recommendation_reason}</p>'
        
        # 각 추천 전공에 대해 상세 계산
        for desired_major in recommended_majors:
            result = calculate_specific_major_recommendation(
                admission_year, 
                primary_major, 
                completed_required, 
                completed_elective, 
                desired_major, 
                data_dict
            )
            response += result
    else:
        response += create_simple_card("<p style='margin:0;'>현재 데이터에서 추천 가능한 전공을 찾을 수 없습니다.</p>", "#fff3e0", "#ff9800")
    
    response += create_tip_box("더 자세한 정보는 각 전공의 교과목과 연락처를 확인해보세요!")
    response += create_contact_box()
    
    return response, "RECOMMENDATION"


def handle_greeting(user_input, extracted_info, data_dict):
    response = create_header_card("안녕하세요!", "👋", "#667eea")
    
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
    response = create_header_card("잠깐만요!", "⚠️", "#ff6b6b")
    response += create_warning_box("부적절한 표현이 감지되었어요.")
    response += create_simple_card("<p style='margin:0;'>다전공 관련 질문을 해주시면 친절하게 답변드릴게요! 😊</p>", "#f0f7ff", "#007bff")
    return response, "BLOCKED"


def handle_out_of_scope(user_input, extracted_info, data_dict):
    response = create_header_card("모릅니다", "🚫", "#636e72")
    
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
    # "선택 안 함"일 때는 표시하지 않음
    if not major or major == "선택 안 함":
        return
    
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
    # 마지막 괄호만 제거 (교육운영전공 부분)
    # 예: "(기계)반도체 부품장비 MD(기계공학전공)" -> "(기계)반도체 부품장비 MD"
    if major.endswith(')') and '(' in major:
        # 마지막 여는 괄호의 위치 찾기
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            clean_major = major[:last_open_paren].strip()
    
    search_keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace('(', '').replace(')', '').replace(' ', '').strip()
    
    type_matched = CURRICULUM_MAPPING[CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image)]
    
    if type_matched.empty:
        return
    
    # 1. 전공명 정확 매칭
    filtered = type_matched[type_matched['전공명'] == clean_major]
    
    # 2. 원본 전공명으로 매칭
    if filtered.empty:
        filtered = type_matched[type_matched['전공명'] == major]
    
    # 3. 공백 제거 후 매칭
    if filtered.empty:
        clean_major_no_space = clean_major.replace(' ', '')
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            cm_major_no_space = cm_major.replace(' ', '')
            if clean_major_no_space == cm_major_no_space:
                filtered = type_matched[type_matched['전공명'] == cm_major]
                break
    
    # 4. 키워드 부분 매칭
    if filtered.empty and len(search_keyword) >= 2:
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            cm_keyword = cm_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace('(', '').replace(')', '').replace(' ', '').strip()
            
            # 더 유연한 매칭: 키워드가 포함되어 있는지 확인
            if len(cm_keyword) >= 2 and len(search_keyword) >= 2:
                if search_keyword in cm_keyword or cm_keyword in search_keyword:
                    filtered = type_matched[type_matched['전공명'] == cm_major]
                    break
    
    # 🔧 수정 #2: 모든 이미지 표시 (여러 개 지원)
    if not filtered.empty:
        images_shown = 0
        missing_files = []
        total_images = len(filtered)
        
        for idx, row in filtered.iterrows():
            filename = row['파일명']
            
            if pd.notna(filename) and str(filename).strip():
                filename_str = str(filename).strip()
                
                # 콤마로 구분된 여러 파일인지 확인
                if ',' in filename_str:
                    # 여러 파일이 하나의 셀에 들어있는 경우
                    file_list = [f.strip() for f in filename_str.split(',')]
                    for file in file_list:
                        image_path = f"{CURRICULUM_IMAGES_PATH}/{file}"
                        if os.path.exists(image_path):
                            if is_fusion:
                                caption = f"{clean_major} 이수체계도"
                            else:
                                caption = f"{clean_major} 과정 안내 ({images_shown + 1})"
                            st.image(image_path, caption=caption)
                            images_shown += 1
                        else:
                            missing_files.append(file)
                else:
                    # 단일 파일
                    image_path = f"{CURRICULUM_IMAGES_PATH}/{filename_str}"
                    
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
                    else:
                        # 없는 파일 목록에 추가
                        missing_files.append(filename_str)
        
        # 없는 파일들을 한 번에 표시
        if missing_files:
            st.warning(f"⚠️ 다음 이미지 파일을 찾을 수 없습니다:")
            for missing_file in missing_files:
                st.caption(f"   • `{CURRICULUM_IMAGES_PATH}/{missing_file}`")
        
        if images_shown == 0 and not missing_files:
            st.caption("📷 이미지 파일 준비 중입니다.")
    else:
        # 매칭 실패 시 정보 표시
        st.info(f"💡 '{major}' 또는 '{clean_major}'에 해당하는 이미지 정보를 curriculum_mapping에서 찾을 수 없습니다.")


# 🔧 수정 #3: 소단위전공 교과목 'XX MD' 패턴으로 검색
def display_courses(major, program_type):
    """과목 정보 표시 - 학년별/학기별/이수구분별 정리"""
    # "선택 안 함"일 때는 표시하지 않음
    if not major or major == "선택 안 함":
        return False
    
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
    
    # 마지막 괄호만 제거 (교육운영전공 부분)
    # 예: "(기계)반도체 부품장비 MD(기계공학전공)" -> "(기계)반도체 부품장비 MD"
    if major.endswith(')') and '(' in major:
        # 마지막 여는 괄호의 위치 찾기
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            clean_major = major[:last_open_paren].strip()
            display_major = clean_major
    
    # 소단위전공과정의 경우 MD를 유지한 채로 검색
    # 1. 정확한 매칭 (MD 포함)
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == clean_major) & 
        (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
    ]
    
    # 🔧 수정 #3: 소단위전공 "XX MD" 패턴으로 검색
    if courses.empty and is_micro:
        # MD를 제거한 키워드로 유사 매칭
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace(' ', '').strip()
        type_matched = COURSES_DATA[COURSES_DATA['제도유형'].apply(match_program_type_for_courses)]
        
        for course_major in type_matched['전공명'].unique():
            cm_str = str(course_major)
            if 'MD' in cm_str or 'md' in cm_str.lower():
                cm_keyword = cm_str.replace('MD', '').replace('md', '').replace(' ', '').strip()
                if len(keyword) >= 2 and len(cm_keyword) >= 2:
                    # 더 유연한 매칭 (첫 2글자 이상 일치)
                    if keyword[:2] in cm_keyword or cm_keyword[:2] in keyword:
                        courses = type_matched[type_matched['전공명'] == course_major]
                        display_major = cm_str
                        break
    
    # 부분 매칭
    if courses.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('(', '').replace(')', '')[:4]
        if keyword:
            courses = COURSES_DATA[
                (COURSES_DATA['전공명'].str.contains(keyword, na=False, regex=False)) & 
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
                                    course_name = row.get('과목명', '')
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    
                                    # 소단위전공과정: 교과목 운영전공 추가
                                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                                    else:
                                        st.write(f"• {course_name} ({credit})")
                        
                        with col2:
                            if not elective.empty:
                                st.markdown("**🟢 전공선택**")
                                for _, row in elective.iterrows():
                                    course_name = row.get('과목명', '')
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    
                                    # 소단위전공과정: 교과목 운영전공 추가
                                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                                    else:
                                        st.write(f"• {course_name} ({credit})")
                        
                        st.divider()
        else:
            # 학년 정보가 없는 경우 (소단위전공과정 등) - 학기별로만 표시
            semesters = sorted([int(s) for s in courses['학기'].unique() if pd.notna(s)])
            
            if semesters:
                for semester in semesters:
                    st.markdown(f"#### 📅 {semester}학기")
                    semester_courses = courses[courses['학기'] == semester]
                    
                    # 이수구분이 있는 경우
                    has_required = not semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)].empty
                    has_elective = not semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)].empty
                    
                    if has_required or has_elective:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            required = semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)]
                            if not required.empty:
                                st.markdown("**🔴 전공필수**")
                                for _, row in required.iterrows():
                                    course_name = row.get('과목명', '')
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    
                                    # 소단위전공과정: 교과목 운영전공 추가
                                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                                    else:
                                        st.write(f"• {course_name} ({credit})")
                        
                        with col2:
                            elective = semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)]
                            if not elective.empty:
                                st.markdown("**🟢 전공선택**")
                                for _, row in elective.iterrows():
                                    course_name = row.get('과목명', '')
                                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                                    
                                    # 소단위전공과정: 교과목 운영전공 추가
                                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                                    else:
                                        st.write(f"• {course_name} ({credit})")
                    else:
                        # 이수구분이 없는 경우 - 전체 과목 표시
                        for _, row in semester_courses.iterrows():
                            course_name = row.get('과목명', '')
                            credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                            
                            # 소단위전공과정: 교과목 운영전공 추가
                            edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                            if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                            else:
                                st.write(f"• {course_name} ({credit})")
                    
                    st.divider()
            else:
                # 학기 정보도 없는 경우 - 전체 과목 표시
                st.markdown("**📚 교과목 목록**")
                for _, row in courses.iterrows():
                    course_name = row.get('과목명', '')
                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                    
                    # 소단위전공과정: 교과목 운영전공 추가
                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                    else:
                        st.write(f"• {course_name} ({credit})")
        
        # 🔧 수정 #4: 전공 연락처 표시
        st.markdown("---")
        display_major_contact(display_major, program_type)
        return True
    else:
        st.info(f"'{display_major}' 교과목 정보가 없습니다.")
        return False


# 🔧 수정 #4: 전공 문의처에 전공명, 위치 추가
def display_major_contact(major, program_type="전공"):
    """전공 연락처 표시 - 전공명, 연락처, 위치 포함"""
    if MAJORS_INFO.empty:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")
        return
    
    # 교육운영전공 추출 (마지막 괄호 안)
    edu_major = None
    clean_major = major
    if major.endswith(')') and '(' in major:
        # "(기계)반도체 부품장비 MD(기계공학전공)" -> edu_major = "기계공학전공", clean_major = "(기계)반도체 부품장비 MD"
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            edu_major = major[last_open_paren+1:-1].strip()
            clean_major = major[:last_open_paren].strip()
    
    clean_major = clean_major.replace(' MD', '').replace('MD', '').strip()
    
    # 소단위전공과정의 경우 교육운영전공으로 검색
    contact_row = pd.DataFrame()
    if edu_major and ("소단위" in program_type or "마이크로" in program_type):
        # 교육운영전공명으로 검색
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == edu_major]
        if contact_row.empty:
            contact_row = MAJORS_INFO[MAJORS_INFO['교육운영전공'] == edu_major]
    
    # 일반적인 검색
    if contact_row.empty:
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == clean_major]
    
    if contact_row.empty:
        # 괄호 제거 후 키워드 추출
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('(', '').replace(')', '')[:4]
        if keyword:
            contact_row = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains(keyword, na=False, regex=False)]
    
    if not contact_row.empty:
        row = contact_row.iloc[0]
        
        # 소단위전공과정의 경우 교육운영전공명 표시
        if "소단위" in program_type or "마이크로" in program_type:
            # 1. 괄호에서 추출한 교육운영전공 사용
            if edu_major:
                major_name = edu_major
            # 2. MAJORS_INFO의 교육운영전공 컬럼 사용
            elif pd.notna(row.get('교육운영전공')) and str(row.get('교육운영전공')).strip():
                major_name = str(row.get('교육운영전공')).strip()
            # 3. 둘 다 없으면 전공명 사용
            else:
                major_name = row.get('전공명', major)
        else:
            major_name = row.get('전공명', major)
        
        phone = row.get('연락처', '')
        location = row.get('사무실위치', row.get('위치', ''))
        
        # 제도 유형에 따라 문의처 제목 동적 변경
        if "소단위" in program_type or "마이크로" in program_type:
            contact_title = "소단위전공과정 문의처"
        else:
            contact_title = f"{program_type} 문의처"
        
        contact_parts = [f"🎓 **전공명**: {major_name}"]
        if pd.notna(phone) and str(phone).strip():
            contact_parts.append(f"📞 **연락처**: {phone}")
        if pd.notna(location) and str(location).strip():
            contact_parts.append(f"📍 **사무실 위치**: {location}")
        
        st.info(f"**📋 {contact_title}**\n\n" + "\n\n".join(contact_parts))
    else:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")


# ============================================================
# 🖥️ 메인 UI
# ============================================================

def main():
    initialize_session_state()
    
    st.title(APP_TITLE)
    
    # 사이드바
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🎓</h1>
            <h3 style='margin-top: 0;'>HKNU 다전공 안내</h3>
        </div>
        """, unsafe_allow_html=True)
        
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내", "FAQ"], 
            icons=["chat-dots-fill", "journal-bookmark-fill", "question-circle-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#0091FF"},
            }
        )
        
        st.divider()
        
        # AI챗봇 소개
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 4px solid #667eea; 
                    padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="color: #333; margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 600;">
                🤖 AI챗봇 소개
            </h4>
            <p style="color: #555; font-size: 0.82rem; margin: 0 0 8px 0; line-height: 1.6;">
                한경국립대학교 다전공 제도에 관한<br>
                궁금한 사항을 AI챗봇이<br>
                친절하게 답변해드립니다!
            </p>
            <p style="color: #999; font-size: 0.7rem; margin: 0; font-style: italic;">
                ⚠️ 본 챗봇은 단순 참고용입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 다전공 제도 소개
        st.markdown("""
        <div style="background-color: #f0f8f5; border-left: 4px solid #11998e; 
                    padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="color: #333; margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 600;">
                📚 다전공 제도란?
            </h4>
            <p style="color: #555; font-size: 0.82rem; margin: 0; line-height: 1.6;">
                주전공 외에 복수, 융합전공 등<br>
                다양한 학위를 취득하여<br>
                융합형 인재로 성장할 수 있도록<br>
                지원하는 제도입니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 학사지원팀 연락처
        st.markdown("""
        <div style="background-color: #fff3e0; border-left: 4px solid #ff9800; 
                    padding: 12px; border-radius: 8px; margin-bottom: 12px;">
            <p style="color: #333; font-size: 0.8rem; margin: 0; line-height: 1.5;">
                📞 <strong>학사지원팀</strong><br>
                <span style="color: #555; font-size: 0.75rem;">031-670-5035</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Powered by 정보 (학사지원팀 아래로)
        st.markdown("""
        <div style="text-align: left; padding: 8px 0;">
            <p style="color: #999; font-size: 0.7rem; margin: 0 0 4px 0;">
                ⚡ Powered by <strong>Gemini 2.0</strong>
            </p>
        """, unsafe_allow_html=True)
        
        if SEMANTIC_ROUTER is not None:
            st.markdown("""
            <p style="color: #aaa; font-size: 0.65rem; margin: 0;">
                🧠 Semantic Router 활성화
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 메인 콘텐츠
    if menu == "AI챗봇 상담":
        st.subheader("💬 AI 상담원과 대화하기")
        
        with st.expander("💡 어떤 질문을 해야 할지 모르겠나요? (클릭)", expanded=False):
            
            def click_question(q):
                st.session_state.chat_history.append({"role": "user", "content": q})
                response_text, res_type = generate_ai_response(q, st.session_state.chat_history[:-1], ALL_DATA)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
                st.rerun()
            
            # 📋 신청 관련
            cols = st.columns([0.5, 6.5])
            with cols[0]:
                st.markdown("""<div style="padding: 8px 0; text-align: right;"><span style="color: #333; font-weight: bold; font-size: 0.9rem;">📋 신청</span></div>""", unsafe_allow_html=True)
            with cols[1]:
                btn_cols = st.columns(5)
                q_apply = [
                    "자격이 뭐야?",
                    "기간은 언제야?",
                    "신청 방법은 뭐야?",
                    "포기 방법은?",
                    "다전공을 변경하려면?",
                ]
                for i, q in enumerate(q_apply):
                    if btn_cols[i].button(q, key=f"qa_{i}", use_container_width=True):
                        click_question(q)
            
            # 📚 제도 관련
            cols = st.columns([0.5, 6.5])
            with cols[0]:
                st.markdown("""<div style="padding: 8px 0; text-align: right;"><span style="color: #333; font-weight: bold; font-size: 0.9rem;">📚 제도</span></div>""", unsafe_allow_html=True)
            with cols[1]:
                btn_cols = st.columns(6)
                q_program = [
                    "다전공이 뭐야?",
                    "복수전공은 뭐야?",
                    "부전공은 뭐야?",
                    "융합전공 알려줘",
                    "마이크로디그리 뭐야?",
                    "복수·부전공 차이는?",
                ]
                for i, q in enumerate(q_program):
                    if btn_cols[i].button(q, key=f"qp_{i}", use_container_width=True):
                        click_question(q)
            
            # 🎓 학점 관련
            cols = st.columns([0.5, 6.5])
            with cols[0]:
                st.markdown("""<div style="padding: 8px 0; text-align: right;"><span style="color: #333; font-weight: bold; font-size: 0.9rem;">🎓 학점</span></div>""", unsafe_allow_html=True)
            with cols[1]:
                btn_cols = st.columns(4)
                q_credit = [
                    "이수 학점 알려줘",
                    "복수전공 몇 학점?",
                    "졸업 요건은?",
                    "제도별 학점 비교",
                ]
                for i, q in enumerate(q_credit):
                    if btn_cols[i].button(q, key=f"qc_{i}", use_container_width=True):
                        click_question(q)
            
            # 🎯 추천 / 📞 연락처
            cols = st.columns([0.5, 6.5])
            with cols[0]:
                st.markdown("""<div style="padding: 8px 0; text-align: right;"><span style="color: #333; font-weight: bold; font-size: 0.9rem;">🎯 📞</span></div>""", unsafe_allow_html=True)
            with cols[1]:
                btn_cols = st.columns(4)
                q_etc = [
                    "저는 2022학번 경영학전공이고, 전필 3학점, 전선 9학점 들었어요. 다전공 추천해주세요",
                    "경영학전공 연락처 알려줘",
                    "응용수학전공 사무실 위치는?",
                    "기계공학전공 교과목은?",
                ]
                for i, q in enumerate(q_etc):
                    if btn_cols[i].button(q, key=f"qe_{i}", use_container_width=True):
                        click_question(q)
        
        st.divider()
        
        for chat in st.session_state.chat_history:
            avatar = "🧑‍🎓" if chat["role"] == "user" else "🤖"
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"], unsafe_allow_html=True)
        
        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI가 답변을 생성 중입니다..."):
                    response_text, res_type = generate_ai_response(prompt, st.session_state.chat_history[:-1], ALL_DATA)
                    st.markdown(response_text, unsafe_allow_html=True)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
            scroll_to_bottom()
    
    elif menu == "다전공 제도 안내":
        st.markdown("""
        <h1 style="font-size: 2rem; margin-bottom: 20px; color: #1f2937;">
            📊 제도 한눈에 비교
        </h1>
        """, unsafe_allow_html=True)
        
        # 🔧 수정 #5: 제도 비교 카드에 졸업요건, 신청자격 추가
        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '')[:50] + '...' if len(info.get('description', '')) > 50 else info.get('description', '-')
                    qual = info.get('qualification', '-')[:30] + '...' if len(str(info.get('qualification', '-'))) > 30 else info.get('qualification', '-')
                    
                    # HTML을 한 줄로 정리하여 렌더링 문제 방지
                    html = f"""<div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-height: 400px; margin-bottom: 12px;"><h3 style="margin: 0 0 8px 0; color: #1f2937; font-size: 1rem;">🎓 {program}</h3><p style="color: #6b7280; font-size: 11px; margin-bottom: 10px; line-height: 1.4;">{desc}</p><hr style="margin: 8px 0; border-top: 1px solid #e5e7eb;"><div style="font-size: 12px; margin-bottom: 8px;"><strong>📖 이수학점</strong><br><span style="font-size: 11px; line-height: 1.6;">• 본전공: {info.get('credits_primary', '-')}<br>• 다전공: {info.get('credits_multi', '-')}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>✅ 신청자격</strong><br><span style="font-size: 11px; color: #4b5563;">{qual}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>🎓 졸업요건</strong><br><span style="font-size: 11px;">졸업인증: {info.get('graduation_certification', '-')}<br>졸업시험: {info.get('graduation_exam', '-')}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>📜 학위표기</strong><br><span style="font-size: 11px; color: #2563eb;">{str(info.get('degree', '-'))[:30]}</span></div><div style="text-align: right; margin-top: 10px;"><span style="font-size: 11px;">난이도: </span><span style="color: #f59e0b;">{info.get('difficulty', '⭐⭐⭐')}</span></div></div>"""
                    st.markdown(html, unsafe_allow_html=True)
        
        st.divider()
        st.subheader("🔍 상세 정보 조회")
        
        prog_keys = list(ALL_DATA['programs'].keys()) if 'programs' in ALL_DATA else []
        selected_program = st.selectbox("제도 선택", prog_keys)
        
        if selected_program:
            info = ALL_DATA['programs'][selected_program]
            
            tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징"])
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**개요**\n\n{info.get('description', '-')}")
                    
                    # 이수학점 - 줄바꿈으로 보기 좋게
                    credits_text = f"""**이수학점**
- 교양: {info.get('credits_general', '-')}
- 원전공: {info.get('credits_primary', '-')}
- 다전공: {info.get('credits_multi', '-')}"""
                    st.markdown(credits_text)
                    
                    # 졸업요건 - 줄바꿈으로 보기 좋게
                    graduation_text = f"""**졸업요건**
- 졸업인증: {info.get('graduation_certification', '-')}
- 졸업시험: {info.get('graduation_exam', '-')}"""
                    st.markdown(graduation_text)
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
                
                # 특수 제도 확인 (부분 문자열 매칭)
                is_special = any(sp in selected_program for sp in ["융합전공", "융합부전공", "소단위", "마이크로"])
                
                # 계열별 전공 그룹화 - available_majors를 직접 사용
                if is_special:
                    # 특수 제도는 계열 구분 없이 전체로 표시
                    category_majors = {"전체": sorted(available_majors.keys())}
                else:
                    # 일반 제도는 계열별로 그룹화
                    category_majors = get_majors_by_category(selected_program)
                
                if selected_program in target_programs:
                    # 특수 제도 (융합전공 등)는 계열 구분 없이 표시
                    if is_special or len(category_majors) <= 1:
                        col_m1, col_m2, col_m3 = st.columns([3, 3, 1.5])
                        with col_m1:
                            all_majors = []
                            for majors in category_majors.values():
                                all_majors.extend(majors)
                            selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(set(all_majors)))
                        with col_m2:
                            # 본전공도 계열별 구분선 방식
                            primary_categories = get_majors_by_category("복수전공")
                            if len(primary_categories) > 1:
                                # 계열별 구분선 포함된 옵션 생성
                                primary_options_with_dividers = ["선택 안 함"]
                                
                                for category in sorted(primary_categories.keys()):
                                    divider = f"━━━━━━ {category} ━━━━━━"
                                    primary_options_with_dividers.append(divider)
                                    
                                    for major in sorted(primary_categories[category]):
                                        primary_options_with_dividers.append(major)
                                
                                my_primary = st.selectbox(
                                    "나의 본전공",
                                    primary_options_with_dividers,
                                    key=f"special_primary_{selected_program}"
                                )
                                
                                # 구분선 선택 시 경고
                                if my_primary and "━━━" in my_primary:
                                    st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                                    my_primary = "선택 안 함"
                            else:
                                primary_list = []
                                if not PRIMARY_REQ.empty:
                                    primary_list = sorted(PRIMARY_REQ['전공명'].unique().tolist())
                                my_primary = st.selectbox("나의 본전공", ["선택 안 함"] + primary_list)
                        with col_m3:
                            admission_year = st.number_input(
                                "📅 본인 학번",
                                min_value=2020,
                                max_value=datetime.now().year,
                                value=datetime.now().year,
                                key=f"special_admission_year_{selected_program}"
                            )
                    else:
                        # 일반 제도 (복수전공, 부전공)는 계열별 구분선 방식
                        # 다전공 선택 - 계열별 구분선 포함
                        major_options_with_dividers = ["선택 안 함"]
                        major_to_category = {}  # 전공명 -> 계열명 매핑
                        
                        for category in sorted(category_majors.keys()):
                            # 계열 구분선 추가
                            divider = f"━━━━━━ {category} ━━━━━━"
                            major_options_with_dividers.append(divider)
                            
                            # 해당 계열의 전공들 추가
                            for major in sorted(category_majors[category]):
                                major_options_with_dividers.append(major)
                                major_to_category[major] = category
                        
                        # 본전공 선택 - 계열별 구분선 포함
                        primary_categories = get_majors_by_category("복수전공")
                        primary_options_with_dividers = ["선택 안 함"]
                        
                        for category in sorted(primary_categories.keys()):
                            # 계열 구분선 추가
                            divider = f"━━━━━━ {category} ━━━━━━"
                            primary_options_with_dividers.append(divider)
                            
                            # 해당 계열의 전공들 추가
                            for major in sorted(primary_categories[category]):
                                primary_options_with_dividers.append(major)
                        
                        # 한 줄에 3개 필드 배치 (학번 칸은 작게)
                        col1, col2, col3 = st.columns([3, 3, 1.5])
                        
                        with col1:
                            selected_major = st.selectbox(
                                f"🎓 이수하려는 {selected_program}",
                                major_options_with_dividers,
                                key=f"major_select_{selected_program}"
                            )
                        
                        with col2:
                            my_primary = st.selectbox(
                                "🏠 나의 본전공",
                                primary_options_with_dividers,
                                key=f"primary_select_{selected_program}"
                            )
                        
                        with col3:
                            admission_year = st.number_input(
                                "📅 본인 학번",
                                min_value=2020,
                                max_value=datetime.now().year,
                                value=datetime.now().year,
                                key=f"admission_year_{selected_program}"
                            )
                        
                        # 구분선을 선택한 경우 경고
                        if selected_major and "━━━" in selected_major:
                            st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            selected_major = None
                        
                        if my_primary and "━━━" in my_primary:
                            st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            my_primary = "선택 안 함"
                        
                else:
                    # 소단위전공과정 등 - 분야별 구분선으로 표시
                    # 분야별 그룹화 및 교육운영전공 정보 저장
                    field_majors = {}
                    major_to_edu_major = {}  # 전공명 -> 교육운영전공명 매핑
                    
                    if not MAJORS_INFO.empty:
                        # MAJORS_INFO에서 소단위전공과정 필터링
                        mask = MAJORS_INFO['제도유형'].apply(lambda x: any(kw in str(x).lower() for kw in ['소단위', '마이크로', 'md']))
                        micro_df = MAJORS_INFO[mask]
                        
                        # 분야 또는 계열 컬럼 사용
                        group_column = None
                        if '분야' in MAJORS_INFO.columns:
                            group_column = '분야'
                        elif '계열' in MAJORS_INFO.columns:
                            group_column = '계열'
                        
                        for _, row in micro_df.iterrows():
                            # 분야/계열 정보
                            if group_column:
                                field = row.get(group_column, '기타')
                                if pd.isna(field) or str(field).strip() == '':
                                    field = '기타'
                                field = str(field).strip()
                            else:
                                field = '전체'
                            
                            major_name = row['전공명']
                            edu_major = row.get('교육운영전공', '')
                            
                            # 표시용 이름 생성 (교육운영전공 포함)
                            if pd.notna(edu_major) and str(edu_major).strip():
                                display_name = f"{major_name}({str(edu_major).strip()})"
                                major_to_edu_major[display_name] = str(edu_major).strip()
                            else:
                                display_name = major_name
                                major_to_edu_major[display_name] = major_name
                            
                            if field not in field_majors:
                                field_majors[field] = []
                            if display_name not in field_majors[field]:
                                field_majors[field].append(display_name)
                    
                    # 분야별 구분선 포함된 옵션 생성
                    if field_majors and len(field_majors) > 1:
                        # 여러 분야가 있을 때만 구분선 표시
                        major_options_with_dividers = ["선택 안 함"]
                        
                        for field in sorted(field_majors.keys()):
                            # 분야 구분선 추가
                            divider = f"━━━━━━ {field} ━━━━━━"
                            major_options_with_dividers.append(divider)
                            
                            # 해당 분야의 전공들 추가
                            for major in sorted(field_majors[field]):
                                major_options_with_dividers.append(major)
                        
                        selected_major = st.selectbox(
                            f"🎓 이수하려는 {selected_program}",
                            major_options_with_dividers,
                            key=f"micro_major_{selected_program}"
                        )
                        
                        # 구분선 선택 시 경고
                        if selected_major and "━━━" in selected_major:
                            st.warning("⚠️ 분야 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            selected_major = None
                    elif field_majors:
                        # 분야가 1개만 있을 때는 구분선 없이 표시
                        all_majors = []
                        for majors in field_majors.values():
                            all_majors.extend(majors)
                        
                        selected_major = st.selectbox(
                            f"🎓 이수하려는 {selected_program}",
                            ["선택 안 함"] + sorted(all_majors),
                            key=f"micro_major_{selected_program}"
                        )
                    else:
                        # 분야 정보가 없으면 전체 목록으로
                        if category_majors and category_majors.get("전체"):
                            all_majors = category_majors["전체"]
                        else:
                            all_majors = sorted(available_majors.keys())
                        
                        if all_majors:
                            selected_major = st.selectbox(
                                f"🎓 이수하려는 {selected_program}",
                                all_majors,
                                key=f"micro_major_{selected_program}"
                            )
                        else:
                            st.warning(f"⚠️ {selected_program}에 해당하는 전공을 찾을 수 없습니다.")
                            selected_major = None
                    
                    my_primary = "선택 안 함"
                    admission_year = datetime.now().year  # 기본값 설정
                
                
                
                if selected_major:
                    if selected_program in target_programs:
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
            else:
                # available_majors가 비어있을 때
                st.warning(f"⚠️ {selected_program}에 해당하는 전공 목록을 찾을 수 없습니다.")
                st.info("💡 데이터 파일에 해당 제도의 전공 정보가 있는지 확인해주세요.")
    
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
# Updated at Mon Dec 29 13:38:35 UTC 2025