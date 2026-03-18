"""
============================================================
🎓 다전공 안내 AI기반 챗봇
============================================================
버전: 4.0 (리팩토링 버전)
주요 변경사항:
1. FAQ 메뉴 삭제 (AI챗봇 상담, 다전공 제도 안내만 유지)
2. faq_mapping.xlsx 기반 FAQ 검색 우선 적용
3. FAQ → Semantic Router → AI Fallback 순서로 처리
4. YAML 설정 파일과 중복 제거 및 정리
5. 다전공 제도 안내 화면 완전 유지
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
import re
import logging
import time
import hashlib

# Google Sheets 로깅
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("⚠️ gspread 패키지가 없습니다. 로깅이 비활성화됩니다.")

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

LINKS = MESSAGES.get('links', {})
ACADEMIC_NOTICE_URL = LINKS.get('academic_notice', "https://www.hknu.ac.kr/kor/562/subview.do")

PATHS = SETTINGS.get('paths', {})
CURRICULUM_IMAGES_PATH = PATHS.get('curriculum_images', "images/curriculum")

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

# ============================================================
# 🔧 디버그 로깅 설정
# ============================================================
# 환경 변수 DEBUG_MODE=true 로 설정하면 디버그 출력 활성화
# Streamlit Cloud: Settings → Secrets에 DEBUG_MODE = "true" 추가
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'

# Streamlit secrets에서도 확인
try:
    if st.secrets.get('DEBUG_MODE', 'false').lower() == 'true':
        DEBUG_MODE = True
except:
    pass

def debug_print(message):
    """디버그 모드일 때만 출력"""
    if DEBUG_MODE:
        print(message)

# ============================================================

SEMANTIC_ROUTER_AVAILABLE = False
Route = None
SemanticRouter = None
GoogleEncoder = None
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


# ============================================================
# 📊 Google Sheets 로깅 시스템
# ============================================================

@st.cache_resource
def init_google_sheets():
    """Google Sheets 초기화"""
    if not GSPREAD_AVAILABLE:
        return None
    
    try:
        if "gcp_service_account" not in st.secrets:
            print("⚠️ Google Sheets 인증 정보가 없습니다")
            return None
        
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(credentials)
        
        sheet_name = st.secrets.get("google_sheets", {}).get("sheet_name", "chatbot_로그")
        
        try:
            sheet = gc.open(sheet_name)
        except gspread.SpreadsheetNotFound:
            sheet = gc.create(sheet_name)
            _init_worksheets(sheet)
        
        print("✅ Google Sheets 연동 성공")
        return sheet
    except Exception as e:
        print(f"⚠️ Google Sheets 초기화 실패: {e}")
        return None


def _init_worksheets(sheet):
    """워크시트 초기화"""
    try:
        # 대화 로그 시트
        try:
            chat_sheet = sheet.worksheet("chat_logs")
        except:
            chat_sheet = sheet.add_worksheet("chat_logs", 1000, 10)
            chat_sheet.append_row([
                "timestamp", "session_id", "user_question", "bot_response", 
                "response_type", "response_time", "page_context"
            ])
        
        # 답변 실패 로그 시트
        try:
            failed_sheet = sheet.worksheet("failed_responses")
        except:
            failed_sheet = sheet.add_worksheet("failed_responses", 1000, 5)
            failed_sheet.append_row([
                "timestamp", "session_id", "user_question", 
                "attempted_response", "failure_reason"
            ])
        
        # 일일 통계 시트
        try:
            stats_sheet = sheet.worksheet("daily_stats")
        except:
            stats_sheet = sheet.add_worksheet("daily_stats", 1000, 6)
            stats_sheet.append_row([
                "date", "session_id", "first_visit", "last_visit", "total_questions"
            ])
    except Exception as e:
        print(f"⚠️ 워크시트 초기화 실패: {e}")


def log_to_sheets(session_id, user_question, bot_response, response_type, response_time=0.0, page_context=""):
    """Google Sheets에 로그 저장"""
    sheet = st.session_state.get('google_sheet')
    if not sheet:
        return
    
    try:
        chat_sheet = sheet.worksheet("chat_logs")
        stats_sheet = sheet.worksheet("daily_stats")
        
        # 대화 로그 추가
        chat_sheet.append_row([
            datetime.now().isoformat(),
            session_id,
            user_question[:500],
            bot_response[:500],
            response_type,
            response_time,
            page_context
        ])
        
        # 일일 통계 업데이트
        today = datetime.now().date().isoformat()
        stats = stats_sheet.get_all_records()
        
        session_row = None
        for idx, row in enumerate(stats, start=2):
            if row.get('date') == today and row.get('session_id') == session_id:
                session_row = idx
                break
        
        if session_row:
            current_count = stats_sheet.cell(session_row, 5).value
            stats_sheet.update_cell(session_row, 4, datetime.now().isoformat())
            stats_sheet.update_cell(session_row, 5, int(current_count or 0) + 1)
        else:
            stats_sheet.append_row([
                today,
                session_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                1
            ])
    except Exception as e:
        print(f"⚠️ 로깅 실패: {e}")


def log_failed_to_sheets(session_id, user_question, attempted_response, failure_reason):
    """답변 실패 로그 저장"""
    sheet = st.session_state.get('google_sheet')
    if not sheet:
        return
    
    try:
        failed_sheet = sheet.worksheet("failed_responses")
        failed_sheet.append_row([
            datetime.now().isoformat(),
            session_id,
            user_question[:500],
            attempted_response[:500],
            failure_reason
        ])
    except Exception as e:
        print(f"⚠️ 실패 로그 저장 실패: {e}")


# 페이지 설정
st.set_page_config(
    page_title="다전공 안내 챗봇",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://www.hknu.ac.kr',
        'Report a bug': 'https://www.hknu.ac.kr',
        'About': "# 한경국립대 다전공 안내 AI기반 챗봇"
    }
)

# CSS 스타일
hide_streamlit_style = """
<style>
    [data-testid="stDecoration"] { display: none !important; }
    footer { display: none !important; }
    .main .block-container { padding-top: 2rem !important; }
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
    
    # 세션 ID 생성
    if 'session_id' not in st.session_state:
        timestamp = datetime.now().isoformat()
        st.session_state.session_id = hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    # Google Sheets 초기화
    if 'google_sheet' not in st.session_state:
        st.session_state.google_sheet = init_google_sheets()
    
    # 스크롤 플래그 초기화
    if 'should_scroll' not in st.session_state:
        st.session_state.should_scroll = False
    
    # ========== 🆕 대화 컨텍스트 저장용 (연속 질문 처리) ==========
    if 'last_mentioned_program' not in st.session_state:
        st.session_state.last_mentioned_program = None  # 마지막 언급된 제도 (복수전공, 융합전공 등)
    
    if 'last_mentioned_entity' not in st.session_state:
        st.session_state.last_mentioned_entity = None   # 마지막 언급된 전공/과정명
    
    if 'last_mentioned_entity_type' not in st.session_state:
        st.session_state.last_mentioned_entity_type = None  # major 또는 microdegree
    
    if 'context_turn_count' not in st.session_state:
        st.session_state.context_turn_count = 0  # 컨텍스트 유지 턴 수 (3턴까지만 유지)


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
def load_faq_mapping():
    """faq_mapping.xlsx 로드"""
    df = load_excel_data('data/faq_mapping.xlsx')
    if df.empty:
        return pd.DataFrame()
    # 필요한 컬럼만 선택하고 NaN 제거
    required_cols = ['faq_id', 'intent', 'program', 'keyword', 'answer']
    if all(col in df.columns for col in required_cols):
        return df[required_cols].dropna(subset=['answer'])
    return pd.DataFrame()


@st.cache_data
def load_majors_info():
    """전공 정보 로드 (복수전공, 부전공 등 일반 전공)"""
    return load_excel_data('data/majors_info.xlsx')


@st.cache_data
def load_microdegree_info():
    """마이크로디그리(소단위전공과정) 정보 로드 - 과정 중심"""
    return load_excel_data('data/microdegree_info.xlsx')


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
FAQ_MAPPING = load_faq_mapping()
MAJORS_INFO = load_majors_info()
MICRODEGREE_INFO = load_microdegree_info()
GRADUATION_REQ = load_graduation_requirements()
PRIMARY_REQ = load_primary_requirements()

ALL_DATA = {
    'programs': PROGRAM_INFO,
    'curriculum': CURRICULUM_MAPPING,
    'courses': COURSES_DATA,
    'faq_mapping': FAQ_MAPPING,
    'majors': MAJORS_INFO,
    'microdegree': MICRODEGREE_INFO,
    'grad_req': GRADUATION_REQ,
    'primary_req': PRIMARY_REQ,
}


# ============================================================
# 📌 조사 제거 함수 (FAQ 매칭 개선용)
# ============================================================

def normalize_for_matching(text):
    """
    FAQ 매칭용 정규화 - 조사를 포함한 변형을 처리
    
    전략: 특정 패턴의 조사만 제거
    - "XXX가 뭐" → "XXX뭐" (가 제거)
    - "XXX는 뭐" → "XXX뭐" (는 제거)
    - "XXX이 뭐" → "XXX뭐" (이 제거)
    """
    import re
    
    # 소문자 변환
    text = text.lower()
    
    # 특수문자 제거
    text = re.sub(r'[?!.,]', '', text)
    
    # 공백 제거
    text = text.replace(' ', '')
    
    # 1단계: 의문사 앞 조사 제거
    # "OO가뭐" → "OO뭐", "OO는뭐" → "OO뭐"
    particles_before_interrogative = ['가', '는', '은', '이', '을', '를']
    interrogatives = ['뭐', '뭔', '무엇', '어떻', '어떤', '언제', '얼마', '몇']
    
    for particle in particles_before_interrogative:
        for interr in interrogatives:
            pattern = f'{particle}{interr}'
            text = text.replace(pattern, interr)
    
    # 2단계: 동사 앞 조사 제거
    # "OO을신청" → "OO신청"
    verbs = ['신청', '취소', '포기', '변경', '하려', '하고', '할수', '해야', '됩니', '알려', '설명']
    for particle in ['을', '를', '이', '가', '은', '는']:
        for verb in verbs:
            pattern = f'{particle}{verb}'
            text = text.replace(pattern, verb)
    
    return text


def check_program_name_only(user_input):
    """
    프로그램명만 입력되었는지 확인
    예: "마이크로디그리", "복수전공", "MD" 등
    
    Returns:
        str: 매칭된 프로그램명 (FAQ의 program 컬럼과 일치하는 값)
        None: 매칭 안됨
    """
    import re
    
    # 정규화: 소문자, 공백/특수문자 제거
    text_clean = re.sub(r'[?!.,\s]', '', user_input.lower())
    
    # 프로그램명 매핑 (입력 가능한 형태 → FAQ program 값)
    program_patterns = {
        # 마이크로디그리 (다양한 표현)
        '마이크로디그리': '마이크로디그리',
        '소단위전공과정': '마이크로디그리',
        '소단위전공': '마이크로디그리',
        '소단위': '마이크로디그리',
        'md': '마이크로디그리',
        '마디': '마이크로디그리',
        # 기본 프로그램들
        '복수전공': '복수전공',
        '복전': '복수전공',
        '부전공': '부전공',
        '부전': '부전공',
        '융합전공': '융합전공',
        '융합부전공': '융합부전공',
        '연계전공': '연계전공',
        '다전공': '다전공',
        '유연학사제도': '유연학사제도',
        '유연학사': '유연학사제도',
        '유연제도': '유연학사제도',
    }
    
    # 정확히 프로그램명만 입력된 경우
    if text_clean in program_patterns:
        return program_patterns[text_clean]
    
    return None


# ============================================================
# 📌 프로그램 키워드 및 인텐트 정의
# ============================================================

PROGRAM_KEYWORDS = {
    '복수전공': ['복수전공', '복전', '복수'],
    '부전공': ['부전공', '부전'],
    '융합전공': ['융합전공', '융합'],
    '융합부전공': ['융합부전공'],
    '연계전공': ['연계전공', '연계'],
    '소단위전공과정': ['소단위전공과정', '소단위전공', '소단위'],
    '마이크로디그리': ['마이크로디그리', '마이크로', 'md', '마디'],
    '다전공': ['다전공'],
    '유연학사제도': ['유연학사제도', '유연학사'],  # 🔧 독립 프로그램으로 분리
}

def find_matching_majors(query_text, majors_df, microdegree_df):
    # 디버그: 반도체 관련 전공 확인
    if DEBUG_MODE:
        debug_print("\n" + "="*60)
        debug_print("🔍 반도체 관련 전공 확인")
        debug_print("="*60)
        if not MAJORS_INFO.empty and '전공명' in MAJORS_INFO.columns:
            semiconductor = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains('반도체', na=False)]
            debug_print(f"반도체 포함 전공: {len(semiconductor)}개")
            for idx, row in semiconductor.iterrows():
                debug_print(f"  - {row['전공명']}")
        else:
            debug_print("MAJORS_INFO가 비어있거나 '전공명' 컬럼이 없습니다!")
        debug_print("="*60 + "\n")

    debug_print(f"\n[DEBUG find_matching_majors] 입력: {query_text}")
    
    query_clean = query_text.replace(' ', '').lower()
    debug_print(f"[DEBUG] query_clean: {query_clean}")
    
    exact_matches = []
    partial_matches = []
    
    # 1. 일반전공에서 검색
    if not majors_df.empty and '전공명' in majors_df.columns:
        debug_print(f"[DEBUG] 일반전공 검색 시작 ({len(majors_df)}개)")
        for idx, row in majors_df.iterrows():
            major_name = str(row.get('전공명', ''))
            major_clean = major_name.replace(' ', '').lower()
            
            # 🔥 괄호 제거: 정규식 사용
            import re
            major_no_paren = re.sub(r'[(\(].*?[)\)]', '', major_clean)
            query_no_paren = re.sub(r'[(\(].*?[)\)]', '', query_clean)
            
            # 디버깅 출력
            if 'ai반도체' in major_clean or '반도체융합' in major_clean:
                debug_print(f"[DEBUG]   검사: {major_name}")
                debug_print(f"[DEBUG]     major_clean: {major_clean}")
                debug_print(f"[DEBUG]     major_no_paren: {major_no_paren}")
                debug_print(f"[DEBUG]     query_clean: {query_clean}")
                debug_print(f"[DEBUG]     query_no_paren: {query_no_paren}")

            # 정확 매칭 (원본)
            if major_clean == query_clean:
                debug_print(f"[DEBUG]   ✅ 정확 매칭: {major_name}")
                candidate = {
                    'name': major_name,
                    'type': 'major',
                    'program_type': row.get('제도유형', ''),
                    'category': row.get('계열', ''),
                    'department': row.get('소속학부', ''),
                    'match_score': len(major_clean),
                    'exact_match': True
                }
                exact_matches.append(candidate)
            
            # 🔥 정확 매칭 (괄호무시)
            elif major_no_paren and major_no_paren == query_no_paren:
                debug_print(f"[DEBUG]   ✅ 정확 매칭(괄호무시): {major_name}")
                candidate = {
                    'name': major_name,
                    'type': 'major',
                    'program_type': row.get('제도유형', ''),
                    'category': row.get('계열', ''),
                    'department': row.get('소속학부', ''),
                    'match_score': len(major_no_paren),
                    'exact_match': True
                }
                exact_matches.append(candidate)

            # 부분 매칭 (원본)
            elif major_clean and len(major_clean) > 2 and major_clean in query_clean:
                debug_print(f"[DEBUG]   부분 매칭: {major_name}")
                candidate = {
                    'name': major_name,
                    'type': 'major',
                    'program_type': row.get('제도유형', ''),
                    'category': row.get('계열', ''),
                    'department': row.get('소속학부', ''),
                    'match_score': len(major_clean),
                    'exact_match': False
                }
                partial_matches.append(candidate)

            # 🔥 부분 매칭 (괄호무시)
            elif major_no_paren and len(major_no_paren) > 2 and major_no_paren in query_no_paren:
                debug_print(f"[DEBUG]   부분 매칭(괄호무시): {major_name}")
                candidate = {
                    'name': major_name,
                    'type': 'major',
                    'program_type': row.get('제도유형', ''),
                    'category': row.get('계열', ''),
                    'department': row.get('소속학부', ''),
                    'match_score': len(major_no_paren),
                    'exact_match': False
                }
                partial_matches.append(candidate)
    
    # 2. 마이크로디그리에서 검색
    if not microdegree_df.empty and '과정명' in microdegree_df.columns:
        debug_print(f"[DEBUG] 마이크로디그리 검색 시작 ({len(microdegree_df)}개)")
        for idx, row in microdegree_df.iterrows():
            course_name = str(row.get('과정명', ''))
            course_clean = course_name.replace(' ', '').lower()
            
            debug_print(f"[DEBUG]   과정명: {course_name} → clean: {course_clean}")
            
            # 정확한 매칭
            if course_clean == query_clean:
                debug_print(f"[DEBUG]   ✅ 마이크로 정확 매칭: {course_name}")
                candidate = {
                    'name': course_name,
                    'type': 'microdegree',
                    'program_type': '소단위전공과정',
                    'category': row.get('계열', ''),
                    'department': row.get('교육운영전공', ''),
                    'match_score': len(course_clean),
                    'exact_match': True
                }
                exact_matches.append(candidate)
            # 전체 과정명 포함
            elif course_clean and course_clean in query_clean:
                debug_print(f"[DEBUG]   마이크로 부분 매칭: {course_name}")
                candidate = {
                    'name': course_name,
                    'type': 'microdegree',
                    'program_type': '소단위전공과정',
                    'category': row.get('계열', ''),
                    'department': row.get('교육운영전공', ''),
                    'match_score': len(course_clean),
                    'exact_match': False
                }
                partial_matches.append(candidate)
            # MD 제거 후 키워드 매칭
            else:
                keyword = course_clean.replace('md', '').strip()
                if keyword and len(keyword) >= 2 and keyword in query_clean:
                    debug_print(f"[DEBUG]   마이크로 키워드 매칭: {course_name} (키워드: {keyword})")
                    candidate = {
                        'name': course_name,
                        'type': 'microdegree',
                        'program_type': '소단위전공과정',
                        'category': row.get('계열', ''),
                        'department': row.get('교육운영전공', ''),
                        'match_score': len(keyword),
                        'exact_match': False
                    }
                    partial_matches.append(candidate)
    else:
        debug_print(f"[DEBUG] ❌ 마이크로디그리 데이터 없음 또는 '과정명' 컬럼 없음")
    
    # 3. 정확한 매칭 우선
    if exact_matches:
        candidates = exact_matches
        debug_print(f"[DEBUG] 정확 매칭 사용: {len(exact_matches)}개")
    else:
        candidates = partial_matches
        debug_print(f"[DEBUG] 부분 매칭 사용: {len(partial_matches)}개")
    
    # 4. 부분 문자열 중복 제거
    if len(candidates) > 1:
        debug_print(f"[DEBUG] 부분 문자열 중복 체크 시작")
        filtered_candidates = []
        
        # 길이 순으로 정렬 (긴 것부터)
        candidates_sorted = sorted(candidates, key=lambda x: len(x['name']), reverse=True)
        
        for i, cand in enumerate(candidates_sorted):
            cand_clean = cand['name'].replace(' ', '').lower()
            
            # 이 후보보다 긴 후보 중에 이 후보를 포함하는 게 있는지 확인
            is_substring = False
            for j in range(i):
                longer_cand_clean = candidates_sorted[j]['name'].replace(' ', '').lower()
                if cand_clean in longer_cand_clean and cand_clean != longer_cand_clean:
                    is_substring = True
                    debug_print(f"[DEBUG]   '{cand['name']}'은(는) '{candidates_sorted[j]['name']}'의 부분 문자열 → 제외")
                    break
            
            if not is_substring:
                filtered_candidates.append(cand)
        
        candidates = filtered_candidates
        debug_print(f"[DEBUG] 부분 문자열 제거 후: {len(candidates)}개")
    
    # 5. 점수 순으로 정렬
    candidates.sort(key=lambda x: (x['match_score'], len(x['name'])), reverse=True)
    
    # 6. 중복 제거 (이름 기준)
    unique_candidates = []
    seen_names = set()
    for cand in candidates:
        if cand['name'] not in seen_names:
            unique_candidates.append(cand)
            seen_names.add(cand['name'])
    
    needs_filtering = len(unique_candidates) > 1
    
    debug_print(f"[DEBUG] 최종 후보: {len(unique_candidates)}개, 필터링 필요: {needs_filtering}")
    
    return unique_candidates, needs_filtering

def check_microdegree_data():
    """마이크로디그리 데이터 확인 (디버그 모드에서만 출력)"""
    global MICRODEGREE_INFO
    
    if not DEBUG_MODE:
        return
    
    debug_print("\n" + "="*60)
    debug_print("🔍 마이크로디그리 데이터 체크")
    debug_print("="*60)
    
    if 'MICRODEGREE_INFO' not in globals():
        debug_print("❌ MICRODEGREE_INFO 전역 변수가 없습니다!")
        return
    
    if MICRODEGREE_INFO.empty:
        debug_print("❌ MICRODEGREE_INFO가 비어있습니다!")
        debug_print("원인: load_microdegree_info() 함수 확인 필요")
        return
    
    debug_print(f"✅ MICRODEGREE_INFO: {len(MICRODEGREE_INFO)}개 과정")
    debug_print(f"\n컬럼: {list(MICRODEGREE_INFO.columns)}")
    
    if '과정명' in MICRODEGREE_INFO.columns:
        debug_print(f"\n과정명 목록:")
        for idx, name in enumerate(MICRODEGREE_INFO['과정명'].head(10), 1):
            debug_print(f"  {idx}. {name}")
    else:
        debug_print("❌ '과정명' 컬럼이 없습니다!")
    
    debug_print("="*60)

def apply_major_filters(candidates, query_text, detected_program=None):
    """제도유형 및 소속학부 필터 적용"""
    if len(candidates) <= 1:
        return candidates
    
    query_clean = query_text.replace(' ', '').lower()
    filtered = candidates.copy()
    
    # 1. 제도유형 필터
    if detected_program:
        program_filtered = [c for c in filtered if detected_program in c.get('program_type', '')]
        if program_filtered:
            filtered = program_filtered
    
    # 2. 융합전공 vs 일반전공 구분
    if len(filtered) > 1:
        has_convergence = any('융합' in c.get('program_type', '') for c in filtered)
        has_regular = any('복수전공' in c.get('program_type', '') or '부전공' in c.get('program_type', '') for c in filtered)
        
        if has_convergence and has_regular:
            if '융합' in query_clean:
                filtered = [c for c in filtered if '융합' in c.get('program_type', '')]
            else:
                filtered = [c for c in filtered if c.get('type') == 'major' and '융합' not in c.get('program_type', '')]
    
    # 3. 소속학부 필터
    if len(filtered) > 1:
        for candidate in filtered:
            dept = candidate.get('department', '')
            if dept and dept.replace(' ', '').lower() in query_clean:
                return [candidate]
            
    # 4. 필터링 후 다시 정렬! (이 2줄 추가)
    filtered.sort(key=lambda x: (x.get('match_score', 0), len(x.get('name', ''))), reverse=True)

    return filtered


def resolve_major_candidate(candidates, query_text):
    """최종 후보 확정"""
    if not candidates:
        return None, None
    
    if len(candidates) == 1:
        return candidates[0]['name'], candidates[0]['type']
    
    # 여러 후보: 첫 번째 반환
    return candidates[0]['name'], candidates[0]['type']

# ============================================================
# 🔍 [신규] 엔티티 추출 시스템
# ============================================================

def extract_entity_from_text(text):
    """
    [디버깅 버전] 텍스트에서 전공/과정 엔티티 추출
    """
    debug_print(f"\n[DEBUG extract_entity_from_text] 입력: {text}")
    
    # MAJORS_INFO, MICRODEGREE_INFO가 전역 변수로 존재하는지 확인
    global MAJORS_INFO, MICRODEGREE_INFO
    
    if 'MAJORS_INFO' not in globals():
        debug_print("[DEBUG] ❌ MAJORS_INFO가 정의되지 않음!")
        return None, None
    
    if 'MICRODEGREE_INFO' not in globals():
        debug_print("[DEBUG] ❌ MICRODEGREE_INFO가 정의되지 않음!")
        return None, None
    
    debug_print(f"[DEBUG] MAJORS_INFO: {len(MAJORS_INFO)}개")
    debug_print(f"[DEBUG] MICRODEGREE_INFO: {len(MICRODEGREE_INFO)}개")
    
    # 1. 매칭 후보 찾기
    candidates, needs_filtering = find_matching_majors(text, MAJORS_INFO, MICRODEGREE_INFO)
    
    debug_print(f"[DEBUG] 후보 개수: {len(candidates)}")
    for i, cand in enumerate(candidates):
        debug_print(f"[DEBUG]   후보 {i+1}: {cand['name']} (타입: {cand['type']}, 점수: {cand.get('match_score', 0)})")
    
    # 2. 필터링 불필요하면 바로 반환
    if not needs_filtering:
        if candidates:
            result = (candidates[0]['name'], candidates[0]['type'])
            debug_print(f"[DEBUG] ✅ 결과 (필터링 불필요): {result}")
            return result
        debug_print(f"[DEBUG] ❌ 후보 없음")
        return None, None
    
    # 3. 필터링 적용
    detected_program = extract_program_from_text(text)
    debug_print(f"[DEBUG] 감지된 제도: {detected_program}")
    
    filtered_candidates = apply_major_filters(candidates, text, detected_program)
    
    debug_print(f"[DEBUG] 필터링 후 개수: {len(filtered_candidates)}")
    for i, cand in enumerate(filtered_candidates):
        debug_print(f"[DEBUG]   필터 후 {i+1}: {cand['name']} (타입: {cand['type']})")
    
    # 4. 최종 후보 확정
    result = resolve_major_candidate(filtered_candidates, text)
    debug_print(f"[DEBUG] ✅ 최종 결과: {result}")
    return result

def detect_course_keywords(text):
    """[STEP 2] 교과목 관련 키워드 감지"""
    text_clean = text.replace(' ', '').lower()
    course_keywords = ['교과목', '과목', '커리큘럼', '수업', '강의', '이수체계도', '교육과정', '뭐들어', '뭐배워']
    return any(kw in text_clean for kw in course_keywords)


def detect_list_keywords(text):
    """[STEP 3-1] 목록 요청 키워드 감지"""
    text_clean = text.replace(' ', '').lower()
    list_keywords = ['목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '무슨과정', '뭐가있어', '뭐있어']
    return any(kw in text_clean for kw in list_keywords)

# Semantic Router용 인텐트 발화 예시
INTENT_UTTERANCES = {
    'APPLY_QUALIFICATION': [
        "신청 자격이 어떻게 되나요?", "지원 자격 알려주세요", "누가 신청할 수 있어요?",
        "자격 요건이 뭐예요?", "나도 신청 가능해?", "몇 학년부터 할 수 있어요?",
        "조건이 어떻게 돼?", "신청 조건 알려줘", "자격이 뭐야?",
    ],
    'APPLY_PERIOD': [
        "신청 기간이 언제예요?", "언제 신청해요?", "마감일이 언제야?",
        "지원 기간 알려주세요", "언제까지 신청할 수 있어요?", "접수 기간이 어떻게 돼?",
        "몇 월에 신청해?", "기간은 언제야?", "기간 알려줘",
    ],
    'APPLY_METHOD': [
        "신청 방법이 어떻게 되나요?", "어떻게 신청해요?", "신청 절차 알려주세요",
        "지원하려면 어떻게 해야 해?", "신청하는 법 알려줘", "어디서 신청해?",
        "절차가 어떻게 돼?", "방법 알려줘",
    ],
    'APPLY_CANCEL': [
        "포기하고 싶어요", "취소 방법 알려주세요", "철회하려면 어떻게 해?",
        "그만두고 싶어", "포기 신청 어떻게 해?", "취소할 수 있어?",
    ],
    'APPLY_CHANGE': [
        "변경하고 싶어요", "전공 바꾸고 싶어", "수정할 수 있나요?",
        "전환하려면 어떻게 해?", "변경 가능한가요?",
    ],
    'PROGRAM_COMPARISON': [
        "복수전공이랑 부전공 차이가 뭐야?", "뭐가 다른 거야?", "차이점 알려줘",
        "비교해줘", "뭐가 더 좋아?", "어떤 게 나을까?",
    ],
    'PROGRAM_INFO': [
        "복수전공이 뭐야?", "부전공이 뭔가요?", "융합전공 설명해줘",
        "마이크로디그리가 뭐예요?", "다전공이 뭐야?", "다전공 제도가 뭐야?",
    ],
    'CREDIT_INFO': [
        "학점이 몇 학점이야?", "이수학점 알려줘", "졸업하려면 몇 학점 필요해?",
        "전필 몇 학점이야?", "필요한 학점 수",
    ],
    'PROGRAM_TUITION': [
        "등록금이 추가되나요?", "수강료 더 내야 해?", "학비가 올라가?",
        "추가 등록금 있어?", "장학금 받을 수 있어?",
    ],
    'COURSE_SEARCH': [
        "어떤 과목 들어야 해?", "커리큘럼 알려줘", "수업 뭐 들어?",
        "과목 리스트 보여줘", "교과목 알려줘",
    ],
    'CONTACT_SEARCH': [
        "연락처 알려줘", "전화번호가 뭐야?", "문의 어디로 해?",
        "사무실 어디야?", "담당자 연락처",
    ],
    'MAJOR_SEARCH': [ 
        "마이크로디그리 전공 목록", "마이크로디그리 전공 리스트 알려줘", "소단위전공과정 전공 리스트",
    ],
    'MAJOR_INFO': [
        "경영학전공 알려줘", "경영학전공이 뭐야?", "경영학과 설명해줘",
        "소프트웨어융합전공 소개", "소프트웨어융합전공 어떤 전공이야?", "소프트웨어융합전공은 어떤 곳이야?",
        "기계공학전공 어때?", "기계공학전공 정보", "기계공학전공 알려줘",
        "전자공학전공 소개", "전자공학전공이가 뭐야?", "전자공학전공 설명",
        "건축학전공 알려줘", "건축학전공이 뭐야?", "건축학전공 소개",
        "경영학전공 알려줘", "경영학전공이 뭐야?", "경영학전공 소개",
        "응용생명과학전공 어때?", "응용생명과학전공 정보", "응용생명과학전공 설명",
        "화학공학전공 알려줘", "화학공학전공이 뭐야?", "화학공학 소개",
        "법학전공 정보", "법학전공 알려줘", "법학전공은 어떤 곳이야?",
    ],
    'RECOMMENDATION': [
        "뭐가 좋을까?", "추천해줘", "어떤 게 좋아?", "나한테 맞는 거 뭐야?",
        "뭐 해야 할까?", "선택 도와줘",
    ],
    'GREETING': [
        "안녕", "안녕하세요", "하이", "hello", "hi", "반가워",
    ],
}

BLOCKED_KEYWORDS = ['시발', '씨발', 'ㅅㅂ', '병신', 'ㅂㅅ', '지랄', 'ㅈㄹ', '개새끼', '꺼져', '닥쳐', '죽어', '미친', '존나', 'fuck']


# ============================================================
# 🧠 Semantic Router 초기화
# ============================================================

@st.cache_resource
def initialize_semantic_router():
    if not SEMANTIC_ROUTER_AVAILABLE or not SEMANTIC_ROUTER_ENABLED:
        return None
    if Route is None or SemanticRouter is None or GoogleEncoder is None:
        return None
    try:
        encoder = GoogleEncoder(
            name="models/text-embedding-004",
            api_key=st.secrets["GEMINI_API_KEY"]

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
# 🔍 FAQ 매칭 시스템
# ============================================================

def extract_program_from_text(text):
    """텍스트에서 프로그램(제도) 추출"""
    text_lower = text.lower().replace(' ', '')
    
    PROGRAM_KEYWORDS = {
        '복수전공': ['복수전공', '복전', '복수'],
        '부전공': ['부전공', '부전'],
        '융합전공': ['융합전공', '융합'],
        '융합부전공': ['융합부전공'],
        '연계전공': ['연계전공', '연계'],
        '소단위전공과정': ['소단위전공과정', '소단위전공', '소단위'],
        '마이크로디그리': ['마이크로디그리', '마이크로', 'md', '마디'],
        '다전공': ['다전공'],
        '유연학사제도': ['유연학사제도', '유연학사'],
    }
    
    program_order = ['소단위전공과정', '마이크로디그리', '융합부전공', '융합전공', '복수전공', '부전공', '연계전공', '다전공', '유연학사제도']
    
    for program in program_order:
        keywords = PROGRAM_KEYWORDS.get(program, [program])
        for kw in keywords:
            if kw.lower().replace(' ', '') in text_lower:
                return program
    
    if '다전공' in text_lower:
        return '다전공'
    
    return None

def needs_question_completion(user_input, intent, extracted_info, faq_result):
    """
    [개선] 질문 보완이 필요한지 판단
    """
    user_clean = user_input.replace(' ', '').lower()
    
    # 1. 제도 키워드만 있고 구체적 질문 없음 (예: "복수전공")
    program_only_keywords = ['복수전공', '부전공', '융합전공', '마이크로디그리', '소단위전공과정']
    is_program_only = any(kw in user_clean for kw in program_only_keywords) and len(user_clean) < 15
    
    # 2. 리스트/목록 질문에서 대상 누락
    list_keywords = ['목록', '리스트', '종류', '어떤', '무슨', '뭐가있어', '뭐있어']
    if any(kw in user_clean for kw in list_keywords):
        if not extracted_info.get('program'):
            return True, 'target_missing'
    
    # 3. 🔥 신청 관련 키워드만 있고 제도 타입 없음
    intent_only_keywords = {
        '기간': ['기간', '언제', '마감', '일정'],
        '자격': ['자격', '조건', '대상'],
        '방법': ['방법', '어떻게', '절차'],
        '학점': ['학점', '몇학점', '이수'],
    }
    
    for category, keywords in intent_only_keywords.items():
        if any(kw in user_clean for kw in keywords):
            # 신청/기간/자격/방법 등의 키워드는 있지만 제도나 전공이 없음
            target = extracted_info.get('entity') or extracted_info.get('program')
            if not target:
                return True, 'target_missing'
    
    # 4. FAQ 매핑 결과 분석
    if faq_result:
        if isinstance(faq_result, list) and len(faq_result) > 1:
            return True, 'intent_missing'
    
    # 5. target과 intent 분석
    target = extracted_info.get('entity') or extracted_info.get('major') or extracted_info.get('program')
    
    has_intent = False
    for category, keywords in intent_only_keywords.items():
        if any(kw in user_clean for kw in keywords):
            has_intent = True
            break
    
    # target은 있는데 intent가 없는 경우 (예: "경영학전공 알려줘")
    if target and not has_intent and not is_program_only:
        # 이 경우는 전공 정보 요청이므로 보완 불필요
        return False, None
    
    # intent는 있는데 target이 없는 경우 (예: "신청 기간은?")
    if has_intent and not target:
        return True, 'target_missing'
    
    return False, None


def complete_question_with_ai(user_input, previous_question=None, chat_history=None):
    """AI를 사용하여 질문 보완 (대화 이력 활용)"""
    try:
        context = ""
        
        # 대화 이력이 있으면 사용 (최근 3턴만)
        if chat_history:
            recent_history = chat_history[-6:]  # 최근 3턴(6개 메시지)
            context = "\n\n[이전 대화 내용]\n"
            for msg in recent_history:
                role = "학생" if msg["role"] == "user" else "챗봇"
                content = msg["content"]
                # HTML 태그 제거
                content_clean = re.sub(r'<[^>]+>', '', content)
                content_clean = content_clean.strip()[:150]  # 최대 150자
                context += f"{role}: {content_clean}\n"
        elif previous_question:
            # 대화 이력이 없으면 이전 질문만 사용
            context = f"\n\n[이전 질문]\n{previous_question}\n"
        
        prompt = f"""당신은 대학 다전공 안내 챗봇입니다.
학생의 질문이 불완전할 때, 문맥을 파악하여 질문을 보완해주세요.

{context}
[현재 질문]
{user_input}

[지침]
1. 이전 대화에서 언급된 전공명이나 제도명을 파악하세요
2. "그럼", "그거", "그건" 같은 지시어가 있으면 이전 대화 내용을 참고하세요
3. 질문에서 빠진 정보(전공명, 제도명)를 이전 대화에서 찾아 보완하세요
4. 보완된 완전한 질문을 한 문장으로 출력하세요
5. 추가 설명 없이 질문만 출력하세요

보완된 질문:"""

        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={'temperature': 0.3, 'max_output_tokens': 100}
        )
        
        completed = response.text.strip()
        completed = completed.replace('"', '').replace("'", '').replace('출력:', '').strip()
        
        debug_print(f"[DEBUG] 질문 보완: '{user_input}' → '{completed}'")
        return completed
    except Exception as e:
        logging.error(f"질문 보완 실패: {e}")
        return user_input


def complete_question_with_context(user_input, extracted_info, previous_question=None, chat_history=None):
    """
    [개선] 컨텍스트 기반 질문 보완 (대화 이력 활용)
    """
    user_clean = user_input.replace(' ', '').lower()
    
    # 1. 🔥 신청 관련 질문 (기간/자격/방법) - 제도 타입 누락
    intent_keywords = {
        '기간': ['기간', '언제', '마감'],
        '자격': ['자격', '조건'],
        '방법': ['방법', '어떻게'],
        '학점': ['학점', '몇학점'],
    }
    
    has_intent = False
    intent_type = None
    for category, keywords in intent_keywords.items():
        if any(kw in user_clean for kw in keywords):
            has_intent = True
            intent_type = category
            break
    
    has_target = bool(extracted_info.get('entity') or extracted_info.get('program'))
    
    if has_intent and not has_target:
        # 이전 질문에서 제도/전공 추출
        if previous_question:
            prev_entity, _ = extract_entity_from_text(previous_question)
            prev_program = extract_program_from_text(previous_question)
            
            if prev_entity:
                # 이전에 특정 전공 언급했으면
                return f"{prev_entity} {user_input}"
            elif prev_program:
                # 이전에 제도 언급했으면
                return f"{prev_program} {user_input}"
        
        # 이전 질문이 없거나 추출 실패 시 AI로 보완 (대화 이력 전달)
        return complete_question_with_ai(user_input, previous_question, chat_history)
    
    # 2. 목록 질문에서 제도 타입 누락
    list_keywords = ['목록', '리스트', '종류']
    if any(kw in user_clean for kw in list_keywords):
        if not extracted_info.get('program'):
            if previous_question:
                prev_program = extract_program_from_text(previous_question)
                if prev_program:
                    return f"{prev_program} {user_input}"
            
            return complete_question_with_ai(user_input, previous_question, chat_history)
    
    return user_input

def search_faq_mapping(user_input, faq_df):
    """
    FAQ 매핑 검색
    - 세부 과정명 우선 체크 (코드)
    - 구체적 키워드 매칭 (FAQ 파일)
    - 조사 제거로 매칭 정확도 향상
    """
    if faq_df.empty:
        return None, 0
    
    # 🔧 개선: 조사 제거 정규화 적용
    user_clean = user_input.lower().replace(' ', '')
    user_normalized = normalize_for_matching(user_input)  # 조사 제거된 버전
    
    debug_print(f"[DEBUG FAQ] 원본: '{user_input}'")
    debug_print(f"[DEBUG FAQ] 정규화: '{user_normalized}'")
    
    # STEP 1: 복수 프로그램 감지
    program_keywords = ['복수전공', '부전공', '융합전공', '마이크로전공', '마이크로디그리']
    programs_mentioned = [p for p in program_keywords if p in user_clean]
    
    if len(programs_mentioned) >= 2:
        return None, 0
    
    # STEP 1.5: "목록" 질문 감지
    list_keywords = ['목록', '리스트', '전공은', '어떤전공']
    is_list_query = any(kw in user_clean for kw in list_keywords)

    if is_list_query:
        return None, 0

    # STEP 1.6: 연락처 질문 감지 → 연락처 핸들러에서 처리하도록 스킵
    _contact_guard = ['연락처', '전화번호', '번호', '사무실', '문의처', '팩스']
    if any(kw in user_clean for kw in _contact_guard):
        return None, 0

    # 🔥 STEP 1.7: 세부 전공/과정명 감지 (개선: 가장 긴 것 우선)
    has_specific_entity = False
    
    # 🔧 제도명(프로그램명)은 전공명 체크에서 제외 (FAQ로 처리해야 함)
    PROGRAM_NAMES_TO_EXCLUDE = [
        '유연학사제도', '유연학사', '다전공', '복수전공', '부전공', 
        '융합전공', '융합부전공', '연계전공', '소단위전공과정', '마이크로디그리'
    ]
    
    # 일반 전공명 체크
    if not MAJORS_INFO.empty:
        matched_majors = []
        for _, row in MAJORS_INFO.iterrows():
            major_name = str(row.get('전공명', ''))
            major_clean = major_name.replace(' ', '').lower()
            
            # 🔧 제도명은 제외
            if major_clean in [p.lower() for p in PROGRAM_NAMES_TO_EXCLUDE]:
                continue
            
            if major_clean and len(major_clean) > 3 and major_clean in user_clean:
                matched_majors.append((major_name, len(major_clean)))
        
        # 가장 긴 전공명 선택
        if matched_majors:
            matched_majors.sort(key=lambda x: x[1], reverse=True)
            best_major = matched_majors[0][0]
            debug_print(f"[DEBUG] 일반 전공명 감지: {best_major} → FAQ 스킵")
            has_specific_entity = True
    
    # 🔥 마이크로디그리 세부 과정명 체크 (개선: 가장 긴 것 우선)
    if not has_specific_entity and not MICRODEGREE_INFO.empty and '과정명' in MICRODEGREE_INFO.columns:
        matched_courses = []
        
        for _, row in MICRODEGREE_INFO.iterrows():
            course_name = str(row.get('과정명', ''))
            course_clean = course_name.replace(' ', '').lower()
            keyword = course_clean.replace('md', '').strip()
            
            # 조건 1: 과정명 전체 매칭
            if course_clean and course_clean in user_clean:
                matched_courses.append((course_name, len(course_clean), 'full'))
            # 조건 2: 핵심 키워드(3자 이상) + MD 동시 존재
            elif keyword and len(keyword) >= 3 and keyword in user_clean and 'md' in user_clean:
                matched_courses.append((course_name, len(keyword), 'keyword'))
        
        # 가장 긴 과정명 선택 (점수가 높은 것)
        if matched_courses:
            matched_courses.sort(key=lambda x: x[1], reverse=True)
            best_course = matched_courses[0][0]
            match_type = matched_courses[0][2]
            debug_print(f"[DEBUG] 마이크로 과정명 감지({match_type}): {best_course} → FAQ 스킵")
            has_specific_entity = True
    
    # 세부 엔티티가 있으면 FAQ 스킵
    if has_specific_entity:
        return None, 0
    
    # STEP 3: 프로그램 추출
    detected_program = extract_program_from_text(user_input)
    
    # 학사제도 키워드 감지
    academic_keywords = ['증명서', '학점교류', '교직', '교원자격', '휴학', '복학', '전과', '전공변경', '재입학', '수강신청', '학점인정', '이수구분', '성적처리', '졸업식', '학위수여식', '유예', '졸업유예', '조기졸업', '등록금', '학비', '성적', '학점', '수강내역', '계절학기', '수강철회', '졸업', '장학금', '자유학기제', '성적확인', '성적조회', '학점확인', '수강확인', '이수학점확인', '학사시스템']
    is_academic_system = any(kw in user_clean for kw in academic_keywords)
    
    if is_academic_system and not detected_program:
        detected_program = "학사제도"
    
    if not detected_program:
        return None, 0
    
    # STEP 4: FAQ 필터링
    # user_clean에 명시적으로 언급된 보조 프로그램도 포함 (예: "복수·부전공 차이")
    _all_programs = ['복수전공', '부전공', '융합전공', '융합부전공', '연계전공', '소단위전공과정', '마이크로디그리']
    _secondary = [p for p in _all_programs if p != detected_program and p in user_clean]

    if detected_program == "학사제도":
        program_faq = faq_df[faq_df['program'] == '학사제도']
    elif detected_program == "유연학사제도":
        program_faq = faq_df[faq_df['program'] == '유연학사제도']
    elif detected_program in ['소단위전공과정', '마이크로디그리']:
        program_faq = faq_df[faq_df['program'].isin(['소단위전공과정', '마이크로디그리', '다전공'] + _secondary)]
    elif detected_program == "다전공":
        program_faq = faq_df[faq_df['program'] == '다전공']
    else:
        program_faq = faq_df[faq_df['program'].isin([detected_program, '다전공'] + _secondary)]
    
    if program_faq.empty:
        return None, 0
    
    # STEP 5: 키워드 매칭
    best_match = None
    best_score = 0

    for _, row in program_faq.iterrows():
        keywords = str(row.get('keyword', '')).split(',')
        keywords = [k.strip().lower().replace(' ', '') for k in keywords if k.strip()]

        exclude_kws = str(row.get('exclude_keywords', '')).split(',')
        exclude_kws = [e.strip().lower().replace(' ', '') for e in exclude_kws if e.strip()]

        if any(ex in user_clean for ex in exclude_kws):
            continue

        keyword_matches = 0
        total_keyword_length = 0

        for kw in keywords:
            # 원본과 정규화 버전 모두에서 매칭 시도
            if kw in user_clean or kw in user_normalized:
                keyword_matches += 1
                total_keyword_length += len(kw)

        if keyword_matches == 0:
            continue

        score = keyword_matches * 10 + total_keyword_length

        row_program = str(row.get('program', '')).strip()
        if row_program == detected_program:
            score += 30
        elif row_program == '다전공':
            score += 10

        if score > best_score:
            best_score = score
            best_match = row
            debug_print(f"[DEBUG FAQ] 매칭: {row.get('intent')} (score={score})")

    if best_score >= 20:
        return best_match, best_score

    return None, 0

def generate_conversational_response(faq_answer, user_input, program=None):
    """FAQ 답변을 AI를 통해 대화체로 변환"""
    try:
        prompt = f"""당신은 다전공 안내 AI기반 챗봇입니다. 
다음 정보를 바탕으로 학생에게 친근하고 도움이 되는 대화체로 답변해주세요.

[학생 질문]
{user_input}

[참고 정보]
{faq_answer}

[지침]
1. 친근하고 공손한 말투를 사용하세요 (예: "~요", "~습니다")
2. 핵심 정보를 빠뜨리지 마세요
3. 필요시 이모지를 적절히 사용하세요
4. 너무 길지 않게 간결하게 작성하세요
5. URL은 중복없이 답변하세요
6. 문장 끝마다 줄바꿈 추가
7. 마지막에 추가 질문이 있는지 물어보세요
8. URL은 마크다운 볼드(**나 __)로 감싸지 말고 그대로 작성하세요
"""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={'temperature': 0.7, 'max_output_tokens': 800}
        )
        return response.text.strip()
    except Exception as e:
        # AI 실패 시 원본 반환
        return faq_answer


# ============================================================
# 🎨 HTML 카드 스타일 함수
# ============================================================

def create_header_card(title, emoji="📋", color="#667eea"):
    return f"""<p style="margin: 20px 0 16px 0; font-size: 1.3rem; color: inherit; font-weight: 600;">{emoji} {title}</p>"""


def create_info_card(title, content_list, border_color="#007bff", emoji="📌"):
    items_html = ""
    for item in content_list:
        items_html += f'<p style="margin: 6px 0 6px 20px; font-size: 0.95rem; color: inherit;">• {item}</p>\n'
    return f"""<div style="margin: 12px 0;"><p style="color: inherit; margin: 10px 0 8px 0; font-size: 1rem; font-weight: 600;">{emoji} {title}</p>{items_html}</div>"""


def create_simple_card(content, bg_color="#f0f7ff", border_color="#007bff"):
    return f"""<div style="margin: 12px 0; padding: 0;">{content}</div>"""


def create_step_card(step_num, title, description, color="#007bff"):
    return f"""<div style="display: flex; align-items: flex-start; margin: 12px 0; padding: 12px; background: transparent; border: 1px solid #888; border-radius: 10px;"><div style="background: {color}; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 14px; flex-shrink: 0;">{step_num}</div><div><strong style="color: inherit; font-size: 0.95rem;">{title}</strong><p style="margin: 4px 0 0 0; color: inherit; opacity: 0.8; font-size: 0.9rem;">{description}</p></div></div>"""


def create_tip_box(text, emoji="💡"):
    return f"""<p style="margin: 12px 0; color: inherit; opacity: 0.8; font-size: 0.9rem; font-style: italic;">{emoji} <strong>TIP:</strong> {text}</p>"""


def create_warning_box(text, emoji="⚠️"):
    return f"""<p style="margin: 12px 0; color: #ff6b6b; font-size: 0.9rem; font-weight: 500;">{emoji} {text}</p>"""


def create_contact_box():
    return """<p style="margin: 16px 0 0 0; color: inherit; opacity: 0.8; font-size: 0.9rem;">📞 <strong>문의:</strong> 전공 사무실 또는 학사지원팀 <strong>031-670-5035</strong></p>"""


def create_table_html(headers, rows, colors=None):
    header_html = "".join([f'<th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd; font-weight: 600;">{h}</th>' for h in headers])
    rows_html = ""
    for idx, row in enumerate(rows):
        cells = ""
        for i, cell in enumerate(row):
            cells += f'<td style="padding: 10px; border-bottom: 1px solid #eee;">{cell}</td>'
        rows_html += f"<tr>{cells}</tr>"
    return f'<div style="overflow-x: auto; margin: 16px 0;"><table style="width: 100%; border-collapse: collapse;"><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>'


def format_faq_response_html(answer, program=None):
    """FAQ 답변을 예쁜 HTML로 포맷팅"""
    
    # 🔧 0. URL 주변의 마크다운 볼드 서식(**나 __) 제거
    # AI가 (__URL)를__ 형태로 출력하는 문제 해결
    # 간단하게 __ 와 ** 를 모두 제거 (URL 주변에서만)
    
    # (__URL 형태 → (URL
    answer = re.sub(r'\(__\s*(https?://)', r'(\1', answer)
    # URL)를__ 또는 URL)__ 형태 → URL)를 또는 URL)
    answer = re.sub(r'(\)[\s가-힣]*)__', r'\1', answer)
    # __URL__ 형태 → URL
    answer = re.sub(r'__\s*(https?://[^\s__]+)\s*__', r'\1', answer)
    # **URL** 형태 → URL  
    answer = re.sub(r'\*\*\s*(https?://[^\s*]+)\s*\*\*', r'\1', answer)
    # 남은 독립적인 __ 제거 (URL 근처)
    answer = re.sub(r'__(https?://)', r'\1', answer)
    answer = re.sub(r'(https?://[^\s]+)__', lambda m: m.group(1).rstrip('_'), answer)
    
    # 1. 마크다운 링크 변환 [텍스트](URL) → HTML 링크
    markdown_link_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'
    answer = re.sub(markdown_link_pattern, r'<a href="\2" target="_blank" style="color: #007bff; text-decoration: underline;">\1</a>', answer)
    
    # 2. 남은 일반 URL 변환
    # URL 패턴: http(s)://로 시작, 공백/한글 전까지 (단, URL 내부의 valid 문자는 포함)
    def replace_url(match):
        url = match.group(1)
        # URL 끝에 붙은 구두점/괄호/한글 제거
        trailing = ''
        while url and url[-1] in '.,;:!?)]}」』_*':
            trailing = url[-1] + trailing
            url = url[:-1]
        if url:  # URL이 비어있지 않은 경우만 링크 생성
            return f'<a href="{url}" target="_blank" style="color: #007bff; text-decoration: underline;">{url}</a>{trailing}'
        return match.group(0)
    
    # 이미 <a> 태그 안에 있는 URL은 제외, 한글이 나오면 URL 종료
    plain_url_pattern = r'(?<!href=")(?<!">)(https?://[^\s<>가-힣]+)(?!</a>)'
    answer = re.sub(plain_url_pattern, replace_url, answer)
    
    # 번호 리스트 (1. 2. 3.) 처리
    lines = answer.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 번호 리스트 패턴
        if re.match(r'^\d+\.', line):
            if not in_list:
                formatted_lines.append('<ol style="margin: 10px 0; padding-left: 20px;">')
                in_list = True
            # 번호 제거하고 내용만
            content = re.sub(r'^\d+\.\s*', '', line)
            formatted_lines.append(f'<li style="margin: 5px 0; color: inherit;">{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ol>')
                in_list = False
            formatted_lines.append(f'<p style="margin: 8px 0; color: inherit; line-height: 1.6;">{line}</p>')
    
    if in_list:
        formatted_lines.append('</ol>')
    
    content = '\n'.join(formatted_lines)
    
    # 프로그램별 색상
    colors = {
        '복수전공': '#667eea',
        '부전공': '#11998e',
        '융합전공': '#f093fb',
        '융합부전공': '#4facfe',
        '연계전공': '#fa709a',
        '소단위전공과정': '#a8edea',
        '마이크로디그리': '#a8edea',
        '다전공': '#667eea',
    }
    color = colors.get(program, '#667eea')
    
    return f"""
<div style="background: transparent; border: 1px solid #888; border-left: 4px solid {color}; border-radius: 12px; padding: 16px; margin: 12px 0;">
    {content}
</div>
"""

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


# ============================================================
# 🆕 연속 질문 처리를 위한 컨텍스트 관리 함수들
# ============================================================

def is_followup_question(user_input):
    """
    후속 질문인지 판단
    - 짧은 질문 (10자 이하)
    - 지시어 포함 (그거, 그럼, 그건, 거기, 이건)
    - 제도/전공명 없이 질문만 있는 경우
    
    🔧 수정: 전공명이 포함되어 있으면 후속 질문이 아님!
    """
    user_clean = user_input.replace(' ', '').lower()
    
    # 🆕 0. 먼저 전공명/과정명이 있는지 확인 (있으면 후속 질문 아님!)
    # 전공명 패턴: ~전공, ~학과, ~과정, MD 등
    major_patterns = ['전공', '학과', '과정']
    has_major_name = any(p in user_clean for p in major_patterns)
    
    # 🔧 제도 키워드 확인 (먼저 체크!)
    program_keywords = [
        '복수전공', '부전공', '융합전공', '마이크로', '소단위', '연계전공', 'md',
        '유연학사제도', '유연학사', '다전공'  # 🔧 추가
    ]
    has_program = any(kw in user_clean for kw in program_keywords)
    
    # 🔧 전공명이나 제도명이 있으면 바로 False 반환 (후속 질문 아님)
    if has_major_name or has_program:
        return False
    
    # 1. 지시어 패턴 (명확한 후속 질문 표현)
    followup_indicators = [
        '그거', '그럼', '그건', '그래서', '거기', '이건', '그리고',
        '그러면', '그렇다면', '그전공', '그과정', '거긴', '그곳',
        '위에서', '방금', '아까', '위의'
    ]
    has_indicator = any(ind in user_clean for ind in followup_indicators)
    
    # 2. 질문만 있고 대상이 전혀 없는 패턴 (더 엄격하게)
    question_only_patterns = [
        '신청기간은?', '기간은?', '언제야?', '마감은?',
        '자격은?', '조건은?', '신청자격은?',
        '방법은?', '어떻게해?', '절차는?', '신청방법은?',
        '학점은?', '몇학점?', '이수학점은?',
        '교과목은?', '과목은?',
        '연락처는?', '전화번호는?', '위치는?',
        '차이는?', '뭐가달라?', '똑같아?', '같아?'
    ]
    # 🔧 수정: 정확히 일치하거나 매우 짧은 경우만
    is_question_only = user_input.strip() in question_only_patterns or any(
        user_clean == p.replace('?', '').replace(' ', '') for p in question_only_patterns
    )
    
    # 3. 매우 짧은 질문 (10자 이하) + 제도/전공 키워드 없음
    is_very_short = len(user_clean) <= 10
    
    # 후속 질문 판단 (더 엄격하게)
    if has_indicator:
        return True
    if is_question_only:
        return True
    if is_very_short and not has_program and not has_major_name:
        return True
    
    return False


def get_context_from_session():
    """
    세션에서 이전 대화 컨텍스트 가져오기
    """
    context = {
        'program': st.session_state.get('last_mentioned_program'),
        'entity': st.session_state.get('last_mentioned_entity'),
        'entity_type': st.session_state.get('last_mentioned_entity_type'),
        'turn_count': st.session_state.get('context_turn_count', 0)
    }
    return context


def update_context_in_session(program=None, entity=None, entity_type=None):
    """
    세션에 대화 컨텍스트 업데이트
    """
    if program:
        st.session_state.last_mentioned_program = program
        st.session_state.context_turn_count = 0  # 새 컨텍스트면 카운터 리셋
    
    if entity:
        st.session_state.last_mentioned_entity = entity
        st.session_state.last_mentioned_entity_type = entity_type
        st.session_state.context_turn_count = 0
    
    # 컨텍스트 유지 턴 증가 (매 응답마다)
    st.session_state.context_turn_count = st.session_state.get('context_turn_count', 0) + 1
    
    # 3턴 초과시 컨텍스트 초기화
    if st.session_state.context_turn_count > 3:
        st.session_state.last_mentioned_program = None
        st.session_state.last_mentioned_entity = None
        st.session_state.last_mentioned_entity_type = None


def expand_followup_question(user_input, context):
    """
    후속 질문을 컨텍스트와 결합하여 확장
    """
    program = context.get('program')
    entity = context.get('entity')
    
    user_clean = user_input.replace(' ', '').lower()
    
    # 우선순위: entity > program
    if entity:
        # 교과목 관련 질문
        if any(kw in user_clean for kw in ['교과목', '과목', '커리큘럼', '뭐들어', '뭐배워']):
            return f"{entity} {user_input}"
        # 연락처 관련 질문
        if any(kw in user_clean for kw in ['연락처', '전화번호', '위치', '어디야']):
            return f"{entity} {user_input}"
        # 기타 질문
        return f"{entity} {user_input}"
    
    if program:
        return f"{program} {user_input}"
    
    return user_input


def create_context_guide_message():
    """
    컨텍스트 없을 때 안내 메시지 생성
    """
    return """
<div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid #ff9a56;">
    <p style="margin: 0 0 12px 0; color: #d35400; font-size: 1.1rem; font-weight: 600;">💡 어떤 제도나 전공에 대해 질문하시는 건가요?</p>
    <p style="margin: 8px 0; color: #333; font-size: 0.95rem;">
        더 정확한 답변을 드리기 위해, <strong>제도명</strong>이나 <strong>전공명</strong>을 함께 말씀해 주세요!
    </p>
    <div style="background: white; border-radius: 8px; padding: 12px; margin-top: 12px;">
        <p style="margin: 4px 0; color: #555; font-size: 0.9rem;"><strong>📌 이렇게 질문해 보세요:</strong></p>
        <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #666; font-size: 0.85rem;">
            <li>"<strong>복수전공</strong> 신청 기간은 언제야?"</li>
            <li>"<strong>융합전공</strong> 이수 학점 알려줘"</li>
            <li>"<strong>경영학전공</strong> 연락처 알려줘"</li>
            <li>"<strong>마이크로디그리</strong> 과정 목록 보여줘"</li>
        </ul>
    </div>
</div>
"""


def classify_with_ai(user_input):
    prompt = """당신은 질문 분류 AI입니다. 다음 의도 중 하나로 분류하세요.
[의도]: APPLY_QUALIFICATION, APPLY_PERIOD, APPLY_METHOD, APPLY_CANCEL, APPLY_CHANGE, 
PROGRAM_COMPARISON, PROGRAM_INFO, CREDIT_INFO, PROGRAM_TUITION, COURSE_SEARCH, CONTACT_SEARCH, 
RECOMMENDATION, GREETING, OUT_OF_SCOPE
규칙: 의도 이름만 출력. "다전공이 뭐야?"는 PROGRAM_INFO"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"질문: {user_input}\n\n의도를 분류하세요.",
            config={'system_instruction': prompt, 'temperature': 0, 'max_output_tokens': 50}
        )
        intent = response.text.strip().upper()
        valid_intents = ['APPLY_QUALIFICATION', 'APPLY_PERIOD', 'APPLY_METHOD',
                         'APPLY_CANCEL', 'APPLY_CHANGE', 'PROGRAM_COMPARISON', 'PROGRAM_INFO',
                         'CREDIT_INFO', 'PROGRAM_TUITION', 'COURSE_SEARCH', 'CONTACT_SEARCH',
                         'RECOMMENDATION', 'GREETING', 'OUT_OF_SCOPE']
        for valid in valid_intents:
            if valid in intent:
                return valid
        return 'OUT_OF_SCOPE'
    except:
        return 'OUT_OF_SCOPE'


def classify_intent(user_input, use_ai_fallback=True, chat_history=None):
    """
    [개선] 통합 의도 분류 함수 (대화 컨텍스트 활용 강화)
    - 연속 질문 처리 기능 추가
    """
    debug_print(f"\n[DEBUG classify_intent] 입력: {user_input}")
    
    user_clean = user_input.lower().replace(' ', '')
    
    # 1. 욕설 차단
    BLOCKED_KEYWORDS = ['시발', '씨발', 'ㅅㅂ', '병신', 'ㅂㅅ', '지랄', 'ㅈㄹ', '개새끼', '꺼져', '닥쳐', '죽어', '미친', '존나', 'fuck']
    if any(kw in user_clean for kw in BLOCKED_KEYWORDS):
        debug_print("[DEBUG] ❌ 욕설 차단")
        return 'BLOCKED', 'blocked', {}
    
    # 2. 인사말 처리
    greeting_keywords = ['안녕', '하이', '헬로', 'hello', 'hi', '반가워']
    if any(kw in user_clean for kw in greeting_keywords) and len(user_clean) < 15:
        debug_print("[DEBUG] ✅ 인사말")
        return 'GREETING', 'keyword', {}
    
    # ========== 🆕 3. 후속 질문 처리 ==========
    if is_followup_question(user_input):
        context = get_context_from_session()
        debug_print(f"[DEBUG] 후속 질문 감지! 컨텍스트: program={context['program']}, entity={context['entity']}")
        
        # 컨텍스트가 있고 3턴 이내면 확장
        if (context['program'] or context['entity']) and context['turn_count'] <= 3:
            expanded_input = expand_followup_question(user_input, context)
            debug_print(f"[DEBUG] 확장된 질문: {expanded_input}")
            
            # 확장된 질문으로 재귀 호출 (단, 무한 루프 방지)
            if expanded_input != user_input:
                # 컨텍스트 턴 증가
                update_context_in_session()
                # 🔧 재귀 호출 시 AI fallback 비활성화하여 엉뚱한 추론 방지
                return classify_intent(expanded_input, use_ai_fallback=False, chat_history=chat_history)
        else:
            # 컨텍스트 없으면 안내 필요 플래그 설정
            debug_print("[DEBUG] 컨텍스트 없음 - 안내 메시지 필요")
            return 'NEED_CONTEXT', 'followup', {'original_input': user_input}
    
    # 4. 연락처/전화번호 문의 (최우선)
    contact_keywords = ['연락처', '전화번호', '번호', '문의처', '사무실', '팩스', 'contact', 'call']
    if any(kw in user_clean for kw in contact_keywords):
        debug_print("[DEBUG] ✅ 연락처 문의")
        entity_name, entity_type = extract_entity_from_text(user_input)
        # 🆕 컨텍스트 업데이트
        if entity_name:
            update_context_in_session(entity=entity_name, entity_type=entity_type)
        return 'CONTACT_SEARCH', 'keyword', {'entity': entity_name, 'entity_type': entity_type}
    
    # [STEP 1] 전공/과정 엔티티 추출
    entity_name, entity_type = extract_entity_from_text(user_input)
    debug_print(f"[DEBUG] 엔티티 추출 결과: name={entity_name}, type={entity_type}")
    
    # [STEP 2] 교과목 키워드 감지
    course_keywords = ['교과목', '과목', '커리큘럼', '수업', '강의', '이수체계도', '교육과정', '뭐들어', '뭐배워']
    has_course_keyword = any(kw in user_clean for kw in course_keywords)
    debug_print(f"[DEBUG] 교과목 키워드: {has_course_keyword}")
    
    # [STEP 3] 목록 키워드 감지
    list_keywords = ['목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '무슨과정', '뭐가있어', '뭐있어']
    has_list_keyword = any(kw in user_clean for kw in list_keywords)
    debug_print(f"[DEBUG] 목록 키워드: {has_list_keyword}")
    
    # 제도 유형 추출
    program_type = extract_program_from_text(user_input)
    debug_print(f"[DEBUG] 제도 유형: {program_type}")
    
    # 🆕 컨텍스트 업데이트
    if program_type:
        update_context_in_session(program=program_type)
    if entity_name:
        update_context_in_session(entity=entity_name, entity_type=entity_type)
    
    # 추출된 정보 저장
    extracted_info = {
        'entity': entity_name,
        'entity_type': entity_type,
        'program': program_type,
        'major': entity_name
    }
    
    # 🔧 [수정] AI 질문 보완 로직 제거 - 엉뚱한 추론 방지
    # 대신, 정보가 부족하면 재질문 유도
    
    # 3-1. 특정 전공/과정 + 교과목 키워드 → COURSE_SEARCH
    if entity_name and has_course_keyword:
        debug_print(f"[DEBUG] ✅ 분류: COURSE_SEARCH (엔티티={entity_name}, 교과목 키워드=True)")
        return 'COURSE_SEARCH', 'entity', {
            'entity': entity_name, 
            'entity_type': entity_type,
            'program': program_type,
            'major': entity_name
        }
    
    # 3-2. 제도 유형 + 목록 키워드 → MAJOR_SEARCH (전공 목록)
    if program_type and has_list_keyword:
        debug_print(f"[DEBUG] ✅ 분류: MAJOR_SEARCH (제도={program_type}, 목록 키워드=True)")
        return 'MAJOR_SEARCH', 'keyword', {'program': program_type}
    
    # 3-3. 특정 전공/과정 엔티티만 있음 → MAJOR_INFO (전공 정보)
    if entity_name:
        debug_print(f"[DEBUG] ✅ 분류: MAJOR_INFO (엔티티={entity_name})")
        return 'MAJOR_INFO', 'entity', {
            'entity': entity_name,
            'entity_type': entity_type,
            'program': program_type,
            'major': entity_name
        }
    
    debug_print(f"[DEBUG] ⚠️ 엔티티 없음, 계속 진행...")
    
    # 4. 프로그램 관련 질문 분류 (전역 extract_programs 함수 사용)
    found_programs = extract_programs(user_clean)
    
    if found_programs:
        program = found_programs[0]
        debug_print(f"[DEBUG] 프로그램 발견: {program}")
        if any(kw in user_clean for kw in ['자격', '신청할수있', '조건', '대상', '기준']):
            debug_print(f"[DEBUG] ✅ 분류: APPLY_QUALIFICATION")
            return 'APPLY_QUALIFICATION', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['언제', '기간', '마감', '날짜', '일정', '시기']):
            debug_print(f"[DEBUG] ✅ 분류: APPLY_PERIOD")
            return 'APPLY_PERIOD', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['어떻게', '방법', '절차', '순서', '경로']):
            debug_print(f"[DEBUG] ✅ 분류: APPLY_METHOD")
            return 'APPLY_METHOD', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['학점', '몇학점', '이수학점']):
            debug_print(f"[DEBUG] ✅ 분류: CREDIT_INFO")
            return 'CREDIT_INFO', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['등록금', '수강료', '학비', '장학금']):
            debug_print(f"[DEBUG] ✅ 분류: PROGRAM_TUITION")
            return 'PROGRAM_TUITION', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['취소', '포기', '철회', '그만']):
            debug_print(f"[DEBUG] ✅ 분류: APPLY_CANCEL")
            return 'APPLY_CANCEL', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['변경', '바꾸', '전환']):
            debug_print(f"[DEBUG] ✅ 분류: APPLY_CHANGE")
            return 'APPLY_CHANGE', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['차이', '비교', 'vs']):
            debug_print(f"[DEBUG] ✅ 분류: PROGRAM_COMPARISON")
            return 'PROGRAM_COMPARISON', 'complex', {'program': program}
        debug_print(f"[DEBUG] ✅ 분류: PROGRAM_INFO")
        return 'PROGRAM_INFO', 'inferred', {'program': program}
    
    # 5. Semantic Router
    if SEMANTIC_ROUTER is not None:
        debug_print("[DEBUG] Semantic Router 시도")
        semantic_intent, score = classify_with_semantic_router(user_input)
        if semantic_intent:
            debug_print(f"[DEBUG] ✅ Semantic Router: {semantic_intent}")
            return semantic_intent, 'semantic', extract_additional_info(user_input, semantic_intent)
    
    # 6. AI Fallback
    if use_ai_fallback:
        debug_print("[DEBUG] AI Fallback 시도")
        try:
            ai_intent = classify_with_ai(user_input)
            if ai_intent not in ['OUT_OF_SCOPE', 'BLOCKED']:
                debug_print(f"[DEBUG] ✅ AI 분류: {ai_intent}")
                return ai_intent, 'ai', extract_additional_info(user_input, ai_intent)
        except:
            pass
    
    debug_print("[DEBUG] ❌ 분류: OUT_OF_SCOPE")
    return 'OUT_OF_SCOPE', 'fallback', {}


# ============================================================
# 🏫 계열별 전공 그룹화 헬퍼 함수
# ============================================================

def get_majors_by_category(program_type=None, data_source="majors"):
    """계열별로 전공을 그룹화하여 반환"""
    special_programs = ["융합전공", "융합부전공", "소단위전공과정", "마이크로디그리"]
    
    # 🔥 마이크로디그리/소단위전공과정인 경우 microdegree_info.xlsx 사용
    if program_type in ["소단위전공과정", "마이크로디그리"]:
        if not MICRODEGREE_INFO.empty and '과정명' in MICRODEGREE_INFO.columns:
            field_column = '계열' if '계열' in MICRODEGREE_INFO.columns else None
            
            if field_column:
                field_courses = {}
                for _, row in MICRODEGREE_INFO.iterrows():
                    field = row.get(field_column, '기타')
                    if pd.isna(field) or str(field).strip() == '':
                        field = '기타'
                    field = str(field).strip()
                    
                    course_name = row.get('과정명', '')
                    if course_name:
                        if field not in field_courses:
                            field_courses[field] = []
                        if course_name not in field_courses[field]:
                            field_courses[field].append(course_name)
                
                for field in field_courses:
                    field_courses[field] = sorted(field_courses[field])
                
                return field_courses
            else:
                return {"전체": sorted(MICRODEGREE_INFO['과정명'].unique().tolist())}
        return {}
    
    # 융합전공, 융합부전공
    if program_type in ["융합전공", "융합부전공"]:
        majors_list = []
        
        if not MAJORS_INFO.empty and '제도유형' in MAJORS_INFO.columns:
            if program_type == "융합전공":
                mask = MAJORS_INFO['제도유형'].str.contains('융합전공', na=False) & ~MAJORS_INFO['제도유형'].str.contains('융합부전공', na=False)
            else:
                mask = MAJORS_INFO['제도유형'].str.contains(program_type, na=False)
            majors_list = MAJORS_INFO[mask]['전공명'].unique().tolist()
        
        if data_source == "courses" and not COURSES_DATA.empty and '제도유형' in COURSES_DATA.columns:
            if program_type == "융합전공":
                mask = COURSES_DATA['제도유형'].str.contains('융합전공', na=False) & ~COURSES_DATA['제도유형'].str.contains('융합부전공', na=False)
            else:
                mask = COURSES_DATA['제도유형'].str.contains(program_type, na=False)
            for m in COURSES_DATA[mask]['전공명'].unique():
                if m not in majors_list:
                    majors_list.append(m)
        
        return {"전체": sorted(majors_list)} if majors_list else {}
    
    # 일반 전공 (복수전공, 부전공, 연계전공)
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
            category_majors["전체"] = filtered_df['전공명'].unique().tolist()
    
    for cat in category_majors:
        category_majors[cat] = sorted(category_majors[cat])
    
    return category_majors


def get_category_color(category):
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
# 🎯 핸들러 함수들
# ============================================================

def handle_course_search(user_input, extracted_info, data_dict):
    """
    교과목 검색 최종 완성본
    기능:
    1. 1차(정확) -> 2차(키워드) 검색 유지
    2. 학년/이수구분 빈칸 처리 및 이모티콘 유지
    3. 과목명 옆에 (학점, 교육운영전공) 표시 추가
    """
    courses_data = data_dict.get('courses', pd.DataFrame())
    
    # 데이터 없음 방어 로직
    if courses_data.empty:
        response = create_header_card("교과목 검색", "📚", "#ff6b6b")
        response += create_warning_box("교과목 데이터가 없습니다.")
        response += create_contact_box()
        return response, "ERROR"

    # 1. 엔티티 추출
    entity = extracted_info.get('entity') or extracted_info.get('major')
    entity_type = extracted_info.get('entity_type')
    
    if not entity:
        entity, entity_type = extract_entity_from_text(user_input)
    
    # 2. [1차 검색] 정확한 전공명 매칭
    major_courses = pd.DataFrame()
    if entity:
        major_courses = courses_data[courses_data['전공명'] == entity].copy()
        # 정확한 매칭 없으면 포함 검색 시도
        if major_courses.empty:
            keyword_clean = entity.replace('MD', '').replace('md', '').replace('전공', '').replace(' ', '').strip()
            major_courses = courses_data[courses_data['전공명'].str.contains(keyword_clean, case=False, na=False, regex=False)].copy()

    # 3. [2차 검색 - 기능 유지됨] 일반 키워드 광범위 검색 (Fallback)
    if major_courses.empty:
        search_target = entity if entity else user_input
        # '전공', '과' 등을 떼고 순수 키워드로 검색
        keyword = search_target.replace('전공', '').replace('학과', '').replace('과', '').replace('MD', '').replace('md', '').replace(' ', '').strip()
        
        if keyword:
            major_courses = courses_data[courses_data['전공명'].str.contains(keyword, case=False, na=False, regex=False)].copy()
            # 검색 성공 시 엔티티 이름 업데이트
            if not major_courses.empty and not entity:
                entity = major_courses.iloc[0]['전공명']

    # 검색 실패 시 처리
    if major_courses.empty:
        response = create_header_card("교과목 검색", "📚", "#ff6b6b")
        if entity:
            response += create_warning_box(f"'{entity}' 관련 교과목을 찾을 수 없습니다.")
        else:
            response += create_simple_card("<p style='margin:0;'>찾으시는 전공이나 교과목명을 정확히 입력해주세요.</p>", "#f0f4ff", "#667eea")
        return response, "COURSE_SEARCH"

    # 4. 헤더 및 상단 정보 구성
    actual_name = major_courses['전공명'].iloc[0]
    is_md = (entity_type == 'microdegree') or ('MD' in actual_name) or ('md' in actual_name.lower())
    header_color = "#a8edea" if is_md else "#667eea"
    
    response = create_header_card(f"{actual_name} 교과목", "📚", header_color)
    
    # 부드러운 인사말 추가
    response += f"""
<div style="background: transparent; border: 1px solid #888; border-radius: 8px; padding: 12px; margin: 8px 0;">
    <p style="margin: 0; color: inherit;">{actual_name} 교과목 안내해 드릴게요! 📖</p>
</div>
"""
    
    # 제도유형 표시 (상단 박스)
    info_items = []
    program_types = major_courses['제도유형'].dropna().unique().tolist()
    program_str = ', '.join([str(pt) for pt in program_types])
    if program_str:
        info_items.append(f"📋 <strong>제도유형:</strong> {program_str}")

    if info_items:
        response += '<div style="background: transparent; border: 1px solid #888; border-radius: 8px; padding: 12px; margin: 8px 0; font-size: 0.95em;">'
        for item in info_items:
            response += f'<div style="color: inherit;">{item}</div>'
        response += '</div>'

    # 5. 학년/학기별 리스트 출력 (이모티콘, 빈칸 처리 로직 유지됨)
    emoji_map = {1: "🌱", 2: "🌿", 3: "🌳", 4: "🎓", 999: "♾️"}
    
    # 학년 정렬 (NaN -> 999)
    major_courses['sort_year'] = pd.to_numeric(major_courses['학년'], errors='coerce').fillna(999)
    years = sorted(major_courses['sort_year'].unique())

    for sort_year in years:
        # 학년 표시 텍스트 설정
        if sort_year == 999:
            year_data = major_courses[major_courses['학년'].isna()]
            emoji = emoji_map.get(999)
            year_display = f"{emoji} 학년 무관"
        else:
            year_data = major_courses[major_courses['sort_year'] == sort_year]
            emoji = emoji_map.get(int(sort_year), "📅")
            year_display = f"{emoji} {int(sort_year)}학년"

        if year_data.empty: continue

        response += f"""
<div style="background: transparent; border: 1px solid #888; border-radius: 8px; padding: 16px; margin: 12px 0;">
    <p style="margin: 0 0 12px 0; color: inherit; border-bottom: 2px solid {header_color}; padding-bottom: 8px; font-size: 1.1rem; font-weight: 600;">{year_display}</p>
"""

        # 학기 정렬 (없으면 0으로 처리)
        semesters = sorted([int(s) for s in year_data['학기'].dropna().unique()])
        if not semesters and not year_data.empty: semesters = [0]

        for sem in semesters:
            if sem == 0:
                sem_data = year_data[year_data['학기'].isna()]
                sem_display = "학기 미지정"
            else:
                sem_data = year_data[year_data['학기'] == sem]
                sem_display = f"📆 {sem}학기"
            
            if sem_data.empty: continue

            response += f"""
<div style="margin: 12px 0;">
    <p style="margin: 0 0 8px 0; color: inherit; opacity: 0.9; font-size: 1rem; font-weight: 500;">{sem_display}</p>
"""
            
            # 이수구분 필터링 (빈칸 포함 처리 유지됨)
            mask_required = sem_data['이수구분'].str.contains('필수', na=False)
            mask_elective = sem_data['이수구분'].str.contains('선택', na=False)
            mask_others = ~(mask_required | mask_elective)
            
            required = sem_data[mask_required]
            elective = sem_data[mask_elective]
            others = sem_data[mask_others]

            # 🔥 [수정됨] 과목 리스트 생성 함수: (학점, 교육운영전공) 결합
            def create_course_list(rows, bg_color):
                items = ""
                for _, row in rows.iterrows():
                    course_title = row.get('과목명', '-')
                    
                    # 괄호 안에 들어갈 내용 수집
                    details = []
                    
                    # 1. 학점 정보
                    try:
                        c_val = row.get('학점')
                        if pd.notna(c_val):
                            details.append(f"{int(c_val)}학점")
                    except:
                        pass
                    
                    # 2. 교육운영전공 정보 (컬럼이 있고 값이 있을 때만)
                    if '교육운영전공' in row.index:
                        op_major = row.get('교육운영전공')
                        if pd.notna(op_major):
                            op_str = str(op_major).strip()
                            # 'nan' 문자열이나 빈 문자열이 아닐 때만 추가
                            if op_str and op_str.lower() != 'nan':
                                details.append(op_str)
                    
                    # 3. 괄호 포맷팅: (3학점, 행정학전공)
                    detail_str = f" ({', '.join(details)})" if details else ""
                    
                    # --- 과목개요 ---
                    outline = row.get('교과목개요') if '교과목개요' in row.index else None
                    has_outline = pd.notna(outline) and str(outline).strip() != ""

                    if has_outline:
                        items += f"""
<li style="margin: 4px 0;">
    <details>
        <summary style="cursor: pointer; padding: 6px 10px; border-radius: 4px; color: inherit;">
            • {course_title}{detail_str}
        </summary>
        <div style="margin: 6px 0 0 18px; font-size: 13px; color: inherit; opacity: 0.8;">
            {outline}
        </div>
    </details>
</li>
"""
                    else:
                        items += f"""
<li style="margin: 4px 0; padding: 6px 10px; color: inherit;">
    • {course_title}{detail_str}
</li>
"""
                return items

            # 각 섹션 출력
            if not required.empty:
                response += f"""
<div style="margin: 8px 0;">
    <strong style="color: #dc3545;">🔴 전공필수</strong>
    <ul style="list-style: none; padding-left: 0; margin: 8px 0;">
        {create_course_list(required, "")}
</ul>
</div>"""
            
            if not elective.empty:
                response += f"""
<div style="margin: 8px 0;">
    <strong style="color: #28a745;">🟢 전공선택</strong>
    <ul style="list-style: none; padding-left: 0; margin: 8px 0;">
         {create_course_list(elective, "")}
</ul>
</div>"""
                
            if not others.empty:
                response += f"""
<div style="margin: 8px 0;">
    <strong style="color: #007bff;">🔵 전공/자유</strong>
    <ul style="list-style: none; padding-left: 0; margin: 8px 0;">
         {create_course_list(others, "")}
</ul>
</div>"""
            
            response += """</div>""" 
        
        response += """</div>"""

    # 하단 팁 메시지
    if is_md:
        response += create_tip_box(f"💡 {actual_name}에 대해 더 알고 싶으시면 '{actual_name} 설명해줘'라고 물어보세요!")
    else:
        response += create_tip_box(f"💡 더 자세한 사항이 궁금하시면 왼쪽 메뉴의 '다전공 제도 안내'를 참고해 주세요!")
    
    response += create_contact_box()
    
    return response, "COURSE_SEARCH"

def handle_contact_search(user_input, extracted_info, data_dict):
    """연락처 검색 - 마이크로디그리는 microdegree_info 사용"""
    entity = extracted_info.get('entity') or extracted_info.get('major')
    entity_type = extracted_info.get('entity_type')
    
    majors_info = data_dict.get('majors', MAJORS_INFO)
    microdegree_info = data_dict.get('microdegree', MICRODEGREE_INFO)
    
    # 🔥 엔티티가 없으면 새로 추출
    if not entity:
        entity, entity_type = extract_entity_from_text(user_input)
    
    # 엔티티가 여전히 없으면 안내
    if not entity:
        response = create_header_card("연락처 조회", "📞", "#667eea")
        response += create_simple_card("<p style='margin:0;'>어떤 전공/과정의 연락처를 찾으시나요?</p>", "#f0f4ff", "#667eea")
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
        response += create_tip_box("예시: \"경영학전공 연락처 알려줘\", \"식품품질관리 MD 전화번호\"")
        response += create_contact_box()
        return response, "CONTACT_SEARCH"
    
    # 🔥 마이크로디그리 과정인 경우 - microdegree_info 사용
    if entity_type == 'microdegree' and not microdegree_info.empty:
        keyword = entity.replace('MD', '').replace('md', '').replace(' ', '').strip()
        result = microdegree_info[microdegree_info['과정명'].str.contains(keyword, case=False, na=False, regex=False)]
        
        if not result.empty:
            row = result.iloc[0]
            
            response = create_header_card(f"{row['과정명']} 정보", "📞", "#11998e")
            response += f"""
<div style="background: transparent; border: 1px solid #888; border-left: 4px solid #11998e; border-radius: 8px; padding: 16px; margin: 8px 0;">
    <p style="margin: 0 0 12px 0; color: inherit; opacity: 0.9;">{row['과정명']} 안내해 드릴게요! 😊</p>
    <p style="margin: 8px 0; color: inherit;"><strong>🎓 과정명:</strong> {row['과정명']}</p>
    <p style="margin: 8px 0; color: inherit;"><strong>🏫 교육운영전공:</strong> {row.get('교육운영전공', '-')}</p>
    <p style="margin: 8px 0; color: inherit;"><strong>📱 연락처:</strong> {row.get('연락처', '-')}</p>
    <p style="margin: 8px 0; color: inherit;"><strong>📍 위치:</strong> {row.get('위치', '-')}</p>
</div>
"""
            return response, "CONTACT_SEARCH"
    
    # 🔥 일반 전공인 경우 - majors_info 사용
    if not majors_info.empty:
        keyword = entity.replace('전공', '').replace('(', '').replace(')', '').replace(' ', '').strip()
        result = majors_info[majors_info['전공명'].str.contains(keyword, case=False, na=False, regex=False)]
        
        if not result.empty:
            row = result.iloc[0]
            response = create_header_card(f"{row['전공명']} 정보", "📞", "#11998e")

            response += f"""
<div style="background: transparent; border: 1px solid #888; border-left: 4px solid #11998e; border-radius: 8px; padding: 16px; margin: 8px 0;">
    <p style="margin: 0 0 12px 0; color: inherit; opacity: 0.9;">{row['전공명']} 안내해 드릴게요! 😊</p>
    <p style="margin: 8px 0; color: inherit;"><strong>🎓 전공명:</strong> {row['전공명']}</p>
    <p style="margin: 8px 0; color: inherit;"><strong>📱 연락처:</strong> {row.get('연락처', '-')}</p>
    <p style="margin: 8px 0; color: inherit;"><strong>📍 위치:</strong> {row.get('위치', row.get('사무실위치', '-'))}</p>
"""
            
            homepage = row.get('홈페이지', '-')
            if homepage and homepage != '-' and str(homepage).startswith('http'):
                response += f'    <p style="margin: 8px 0; color: inherit;"><strong>🌐 홈페이지:</strong> <a href="{homepage}" target="_blank" style="color: #e83e8c; text-decoration: none;">{homepage} 🔗</a></p>\n'
            else:
                response += f'    <p style="margin: 8px 0; color: inherit;"><strong>🌐 홈페이지:</strong> {homepage}</p>\n'
            
            response += "</div>"
            return response, "CONTACT_SEARCH"
    
    # 찾지 못한 경우
    response = create_header_card("연락처 조회", "📞", "#ff6b6b")
    response += create_warning_box(f"'{entity}' 연락처를 찾을 수 없습니다.")
    response += create_contact_box()
    return response, "ERROR"


def handle_recommendation(user_input, extracted_info, data_dict):
    year_match = re.search(r'(\d{4})학번', user_input)
    major_match = re.search(r'([가-힣]+전공)', user_input)
    required_match = re.search(r'전필\s*(\d+)학점', user_input)
    elective_match = re.search(r'전선\s*(\d+)학점', user_input)
    
    if not (year_match and major_match and (required_match or elective_match)):
        response = create_header_card("맞춤형 다전공 추천", "🎯", "#f093fb")
        response += create_simple_card("<p style='margin:0; font-size: 0.95rem;'>정확한 추천을 위해 아래 정보가 필요합니다</p>", "#fef0f5", "#f5576c")
        response += create_info_card("필요한 정보", [
            "📅 기준학번 (예: 2022학번)",
            "🎓 현재 본전공 (예: 경영학전공)",
            "📊 이수한 전공필수/전공선택 학점"
        ], "#f093fb", "📋")
        response += create_tip_box("예시: \"2022학번 경영학전공 전필 15학점 전선 12학점 이수했어. 추천해줘\"")
        response += create_contact_box()
        return response, "RECOMMENDATION"
    
    admission_year = int(year_match.group(1))
    primary_major = major_match.group(1)
    completed_required = int(required_match.group(1)) if required_match else 0
    completed_elective = int(elective_match.group(1)) if elective_match else 0
    total_credits = completed_required + completed_elective
    
    response = create_header_card("맞춤형 다전공 추천", "🎯", "#f093fb")
    
    response += create_info_card("입력하신 정보", [
        f"📅 학번: {admission_year}학번",
        f"🎓 본전공: {primary_major}",
        f"📊 이수학점: 전필 {completed_required}학점, 전선 {completed_elective}학점 (총 {total_credits}학점)"
    ], "#667eea", "📋")
    
    # 학점 기준 추천
    if total_credits < 20:
        recommendation = "소단위전공과정(마이크로디그리) 또는 부전공"
        reason = "현재 이수학점이 적어 부담이 적은 제도를 추천드립니다."
    elif total_credits < 40:
        recommendation = "부전공 또는 융합부전공"
        reason = "적절한 학점을 이수하셨습니다. 부전공 도전을 추천드립니다."
    else:
        recommendation = "복수전공 또는 융합전공"
        reason = "충분한 학점을 이수하셨습니다. 복수전공 도전 가능합니다!"
    
    response += f"""
<div style="background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%); border-left: 4px solid #f093fb; border-radius: 12px; padding: 16px; margin: 16px 0;">
    <p style="margin: 0 0 10px 0; color: #f093fb; font-size: 1.1rem; font-weight: 600;">🎯 추천 다전공</p>
    <p style="font-size: 1.1rem; font-weight: bold; color: #333; margin: 8px 0;">{recommendation}</p>
    <p style="color: #666; font-size: 0.9rem; margin: 8px 0;">💡 {reason}</p>
</div>
"""
    
    response += create_tip_box("왼쪽 '다전공 제도 안내' 메뉴에서 상세 정보를 확인하세요!")
    response += create_contact_box()
    
    return response, "RECOMMENDATION"

def handle_major_info(user_input, extracted_info, data_dict):
    """전공/과정 설명 제공 - 마이크로디그리는 microdegree_info 사용"""
    entity = extracted_info.get('entity') or extracted_info.get('major')
    entity_type = extracted_info.get('entity_type')
    
    majors_info = data_dict.get('majors', MAJORS_INFO)
    microdegree_info = data_dict.get('microdegree', MICRODEGREE_INFO)
    
    # 엔티티가 없으면 새로 추출
    if not entity:
        entity, entity_type = extract_entity_from_text(user_input)
    
    # 엔티티가 여전히 없으면 안내
    if not entity:
        response = create_header_card("전공/과정 정보", "🎓", "#667eea")
        response += create_simple_card("<p style='margin:0;'>어떤 전공이나 과정에 대해 알고 싶으신가요?</p>", "#f0f4ff", "#667eea")
        
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
        
        response += create_tip_box("예시: \"경영학전공 알려줘\", \"식품품질관리 MD 소개\"")
        response += create_contact_box()
        return response, "MAJOR_INFO"
    
    # 🔥 마이크로디그리 과정인 경우 - 개선된 검색
    if entity_type == 'microdegree' and not microdegree_info.empty:
        debug_print(f"[DEBUG handle_major_info] 마이크로디그리 검색: {entity}")
        
        result = pd.DataFrame()
        
        # 1차: 정확한 매칭 (대소문자, 띄어쓰기 무시)
        entity_clean = entity.replace(' ', '').lower()
        for idx, row in microdegree_info.iterrows():
            course_name = str(row.get('과정명', ''))
            course_clean = course_name.replace(' ', '').lower()
            
            if course_clean == entity_clean:
                result = microdegree_info.iloc[[idx]]
                debug_print(f"[DEBUG] ✅ 정확 매칭: {course_name}")
                break
        
        # 2차: 과정명이 엔티티를 포함
        if result.empty:
            for idx, row in microdegree_info.iterrows():
                course_name = str(row.get('과정명', ''))
                course_clean = course_name.replace(' ', '').lower()
                
                if entity_clean in course_clean or course_clean in entity_clean:
                    result = microdegree_info.iloc[[idx]]
                    debug_print(f"[DEBUG] ✅ 부분 매칭: {course_name}")
                    break
        
        # 3차: 키워드 검색 (MD 제거)
        if result.empty:
            keyword = entity.replace('MD', '').replace('md', '').replace(' ', '').strip()
            debug_print(f"[DEBUG] 키워드 검색: {keyword}")
            
            # 키워드가 과정명에 포함되는지 확인
            result = microdegree_info[
                microdegree_info['과정명'].apply(
                    lambda x: keyword.lower() in str(x).replace(' ', '').lower()
                )
            ]
            
            if not result.empty:
                debug_print(f"[DEBUG] ✅ 키워드 매칭: {result.iloc[0]['과정명']}")
        
        # 검색 성공
        if not result.empty:
            row = result.iloc[0]
            course_name = row.get('과정명', entity)
            
            response = create_header_card(f"{course_name} 소개", "🎓", "#a8edea")
            
            # 과정 설명
            description = row.get('과정설명', '')
            if description and pd.notna(description):
                response += f"""
<div style="background: white; border-left: 4px solid #a8edea; border-radius: 8px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="color: #11998e; font-weight: 600; margin-bottom: 8px;">📖 과정 소개</div>
    <p style="margin: 0; color: #333; line-height: 1.6;">{description}</p>
</div>
"""
            
            # 기본 정보
            response += f"""
<div style="background: white; border-left: 4px solid #11998e; border-radius: 8px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="color: #11998e; font-weight: 600; margin-bottom: 12px;">ℹ️ 기본 정보</div>
"""
            
            # 소속 계열
            category = row.get('계열', '-')
            if category and category != '-' and pd.notna(category):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>🏛️ 소속:</strong> {category}</p>\n'
            
            # 제도유형
            program_types = row.get('제도유형', '')
            if program_types and pd.notna(program_types):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📋 신청 가능 제도:</strong> {program_types}</p>\n'
            
            # 교육운영전공
            edu_major = row.get('교육운영전공', '')
            if edu_major and pd.notna(edu_major):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>🎓 교육운영전공:</strong> {edu_major}</p>\n'
            
            # 연락처
            contact = row.get('연락처', '-')
            if contact and contact != '-' and pd.notna(contact):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📱 연락처:</strong> {contact}</p>\n'
            
            # 위치
            location = row.get('위치', row.get('사무실위치', '-'))
            if location and location != '-' and pd.notna(location):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📍 위치:</strong> {location}</p>\n'
            
            # 홈페이지
            homepage = row.get('홈페이지', '-')
            if homepage and homepage != '-' and pd.notna(homepage) and str(homepage).startswith('http'):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>🌐 홈페이지:</strong> <a href="{homepage}" target="_blank" style="color: #667eea; text-decoration: none;">{homepage} 🔗</a></p>\n'
            
            response += "</div>"
            
            # 🔥 수정: course_name 사용, 마이크로디그리 전용 팁
            response += create_tip_box(f"💡 {course_name}의 교과목이 궁금하시면 '{course_name} 교과목 알려줘'라고 물어보세요!")
            response += create_contact_box()
            
            return response, "MAJOR_INFO"
        
        # 🔥 마이크로디그리에서 찾지 못한 경우
        else:
            response = create_header_card("전공/과정 정보", "🎓", "#ff6b6b")
            response += create_warning_box(f"'{entity}' 정보를 찾을 수 없습니다.")
            response += create_contact_box()
            return response, "ERROR"
    
    # 🔥 일반 전공인 경우 - majors_info 사용
    if not majors_info.empty:
        search_keyword = entity.replace('전공', '').replace('과', '').replace('(', '').replace(')', '').replace(' ', '').strip()
        result = majors_info[majors_info['전공명'].str.contains(search_keyword, case=False, na=False, regex=False)]
        
        if not result.empty:
            row = result.iloc[0]
            major_name = row['전공명']
            
            response = create_header_card(f"{major_name} 소개", "🎓", "#667eea")
            
            # 전공 설명
            description = row.get('전공설명', row.get('설명', '-'))
            if description and description != '-' and pd.notna(description):
                response += f"""
<div style="background: white; border-left: 4px solid #667eea; border-radius: 8px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="color: #667eea; font-weight: 600; margin-bottom: 8px;">📖 전공 소개</div>
    <p style="margin: 0; color: #333; line-height: 1.6;">{description}</p>
</div>
"""
            
            # 기본 정보
            response += f"""
<div style="background: white; border-left: 4px solid #11998e; border-radius: 8px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <div style="color: #11998e; font-weight: 600; margin-bottom: 12px;">ℹ️ 기본 정보</div>
"""
            
            # 소속 계열
            category = row.get('계열', '-')
            if category and category != '-' and pd.notna(category):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>🏛️ 소속:</strong> {category}</p>\n'
            
            # 제도유형
            program_types = row.get('제도유형', '')
            if program_types and pd.notna(program_types):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📋 신청 가능 제도:</strong> {program_types}</p>\n'
            
            # 연락처
            contact = row.get('연락처', '-')
            if contact and contact != '-' and pd.notna(contact):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📱 연락처:</strong> {contact}</p>\n'
            
            # 위치
            location = row.get('위치', row.get('사무실위치', '-'))
            if location and location != '-' and pd.notna(location):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>📍 위치:</strong> {location}</p>\n'
            
            # 홈페이지
            homepage = row.get('홈페이지', '-')
            if homepage and homepage != '-' and pd.notna(homepage) and str(homepage).startswith('http'):
                response += f'    <p style="margin: 8px 0; color: #333;"><strong>🌐 홈페이지:</strong> <a href="{homepage}" target="_blank" style="color: #667eea; text-decoration: none;">{homepage} 🔗</a></p>\n'
            
            response += "</div>"
            
            # 🔥 일반 전공용 팁
            response += create_tip_box(f"💡 {major_name}을(를) 복수전공/부전공으로 신청하고 싶으시다면 '복수전공 신청 방법'을 물어보세요!")
            response += create_contact_box()
            
            return response, "MAJOR_INFO"
    
    # 찾지 못한 경우
    response = create_header_card("전공/과정 정보", "🎓", "#ff6b6b")
    response += create_warning_box(f"'{entity}' 정보를 찾을 수 없습니다.")
    response += create_contact_box()
    return response, "ERROR"


def handle_major_search(user_input, extracted_info, data_dict):
    """전공/과정 검색 및 목록 제공 - 마이크로디그리는 microdegree_info 사용"""
    majors_info = data_dict.get('majors', MAJORS_INFO)
    microdegree_info = data_dict.get('microdegree', MICRODEGREE_INFO)
    
    # 프로그램 추출
    program = extracted_info.get('program') or extract_program_from_text(user_input)
    
    # 🔥 마이크로디그리 목록 요청 - microdegree_info 사용
    if program in ['소단위전공과정', '마이크로디그리'] or '마이크로디그리' in user_input.lower() or 'md' in user_input.lower():
        if not microdegree_info.empty and '과정명' in microdegree_info.columns:
            response = create_header_card("소단위전공과정(마이크로디그리) 목록", "📚", "#a8edea")
            
            # 계열별 그룹화
            category_courses = get_majors_by_category("마이크로디그리")
            
            if category_courses:
                response += format_majors_by_category_html(category_courses)
            else:
                response += """
<div style="background: white; border-radius: 8px; padding: 16px; margin: 8px 0;">
    <p style="margin-bottom: 12px; color: #555;">마이크로디그리 과정 목록입니다:</p>
    <ul style="list-style: none; padding: 0;">
"""
                for _, row in microdegree_info.iterrows():
                    course_name = row.get('과정명', '')
                    description = row.get('과정설명', '')
                    if description and pd.notna(description) and len(str(description)) > 50:
                        description = str(description)[:50] + "..."
                    
                    response += f"""
        <li style="margin: 8px 0; padding: 12px; background: #f8f9fa; border-radius: 6px;">
            <strong style="color: #11998e;">• {course_name}</strong>
            {f'<br><span style="font-size: 0.9em; color: #666;">{description}</span>' if description and pd.notna(description) else ''}
        </li>
"""
                response += """
    </ul>
</div>
"""
            
            response += create_tip_box("💡 각 과정에 대해 더 알고 싶으시다면 '식품품질관리 MD 알려줘'처럼 과정명을 물어보세요!")
            response += create_contact_box()
            
            return response, "MAJOR_SEARCH"
    
    # 일반 전공 검색 (복수전공, 부전공 등)
    if program:
        category_majors = get_majors_by_category(program)
        
        if category_majors:
            response = create_header_card(f"{program} 전공 목록", "📚", "#667eea")
            response += format_majors_by_category_html(category_majors)
            response += create_contact_box()
            return response, "MAJOR_SEARCH"
    
    # 전체 전공 목록
    response = create_header_card("전공 목록", "📚", "#667eea")
    category_majors = get_majors_by_category()
    
    if category_majors:
        response += format_majors_by_category_html(category_majors)
    else:
        response += create_warning_box("전공 목록을 불러올 수 없습니다.")
    
    response += create_contact_box()
    
    return response, "MAJOR_SEARCH"

def handle_greeting(user_input, extracted_info, data_dict):
    response = create_header_card("안녕하세요!", "👋", "#667eea")
    response += create_simple_card("<p style='margin:0; font-size: 1rem;'><strong>다전공 안내 AI기반 챗봇</strong>입니다 😊</p>", "#f0f4ff", "#667eea")
    
    response += """
<div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
    <p style="margin: 0 0 12px 0; color: #333; font-size: 1.1rem; font-weight: 600;">🎯 무엇을 도와드릴까요?</p>
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
    response = create_header_card("범위 외 질문", "🚫", "#636e72")
    response += create_simple_card("<p style='margin:0;'>저는 <strong>다전공 안내 AI기반 챗봇</strong>이에요.</p>", "#f8f9fa", "#6c757d")
    
    response += """
<div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
    <p style="margin: 0 0 12px 0; color: #333; font-size: 1.1rem; font-weight: 600;">💬 이런 질문은 답변할 수 있어요!</p>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.9rem;">
        <div style="padding: 8px; background: #e3f2fd; border-radius: 6px;">📝 다전공 신청 기간 알려줘</div>
        <div style="padding: 8px; background: #e3f2fd; border-radius: 6px;">📝 복수전공이 뭐야</div>
        <div style="padding: 8px; background: #e8f5e9; border-radius: 6px;">📊 융합전공, 융합부전공 비교해줘</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📖 응용수학전공 소개해줘</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📖 전자공학전공 교과목 알려줘</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📞 경영학전공 연락처 뭐야?</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📖 마이크로디그리 과정 목록 알려줘</div>
        <div style="padding: 8px; background: #fce4ec; border-radius: 6px;">📖 반려동물 MD 교과목 알려줘</div>
</div>
</div>
"""
    
    response += create_tip_box("위의 <strong>'💡 어떤 질문을 해야 할지 모르겠나요?'</strong>를 클릭해보세요!")
    
    return response, "OUT_OF_SCOPE"


def handle_general(user_input, extracted_info, data_dict):
    return f"죄송합니다. 답변을 생성하지 못했습니다.\n{CONTACT_MESSAGE}", "ERROR"


def handle_need_context(user_input, extracted_info, data_dict):
    """
    🆕 컨텍스트가 필요한 후속 질문에 대한 안내
    """
    original_input = extracted_info.get('original_input', user_input)
    
    response = create_context_guide_message()
    
    # 사용자의 원래 질문도 표시
    response += f"""
<div style="background: #f8f9fa; border-radius: 8px; padding: 12px; margin-top: 8px;">
    <p style="margin: 0; color: #666; font-size: 0.85rem;">
        💬 입력하신 질문: "<strong>{original_input}</strong>"
    </p>
</div>
"""
    
    response += create_contact_box()
    
    return response, "NEED_CONTEXT"


# 핸들러 매핑 (FAQ로 처리되지 않는 경우 사용)
FALLBACK_HANDLERS = {
    'COURSE_SEARCH': handle_course_search,
    'CONTACT_SEARCH': handle_contact_search,
    'MAJOR_SEARCH': handle_major_search,
    'MAJOR_INFO': handle_major_info,
    'RECOMMENDATION': handle_recommendation,
    'GREETING': handle_greeting,
    'BLOCKED': handle_blocked,
    'OUT_OF_SCOPE': handle_out_of_scope,
    'GENERAL': handle_general,
    'NEED_CONTEXT': handle_need_context,  # 🆕 추가
}


# ============================================================
# 🤖 통합 응답 생성 함수
# ============================================================

def save_previous_question(user_input):
    """세션 상태에 이전 질문 저장"""
    if 'previous_question' not in st.session_state:
        st.session_state.previous_question = None
    
    st.session_state.previous_question = user_input
    
def generate_ai_response(user_input, chat_history, data_dict):
    """
    통합 응답 생성 함수
    1. 후속 질문이면 컨텍스트로 확장
    2. FAQ 매핑 검색
    3. 특수 핸들러 (연락처, 과목검색 등)
    4. AI Fallback
    """
    start_time = time.time()
    faq_df = data_dict.get('faq_mapping', FAQ_MAPPING)
    
    # ========== 🔧 핵심 수정: 맨 처음에 질문 확장 ==========
    original_input = user_input  # 원본 보관 (로깅용)
    
    if is_followup_question(user_input):
        context = get_context_from_session()
        debug_print(f"[DEBUG] 후속 질문 감지: {user_input}")
        debug_print(f"[DEBUG] 컨텍스트: program={context['program']}, entity={context['entity']}")
        
        # 🔧 새로운 엔티티가 있으면 후속 질문 확장 안 함
        new_entity, new_entity_type = extract_entity_from_text(user_input)
        if new_entity:
            debug_print(f"[DEBUG] 새로운 엔티티 감지: {new_entity} → 컨텍스트 확장 안 함")
            # 컨텍스트를 새 엔티티로 업데이트
            update_context_in_session(entity=new_entity, entity_type=new_entity_type)
        elif (context['program'] or context['entity']) and context['turn_count'] <= 3:
            user_input = expand_followup_question(user_input, context)
            debug_print(f"[DEBUG] 확장된 질문: {user_input}")
            update_context_in_session()  # 턴 카운트 증가
        else:
            # 컨텍스트 없으면 안내 메시지
            debug_print("[DEBUG] 컨텍스트 없음 - 안내 메시지")
            response = create_context_guide_message()
            response += f"""
<div style="background: #f8f9fa; border-radius: 8px; padding: 12px; margin-top: 8px;">
    <p style="margin: 0; color: #666; font-size: 0.85rem;">
        💬 입력하신 질문: "<strong>{original_input}</strong>"
    </p>
</div>
"""
            response += create_contact_box()
            log_to_sheets(
                st.session_state.get('session_id', 'unknown'),
                original_input, response, 'need_context', 
                time.time() - start_time,
                st.session_state.get('page', 'AI챗봇 상담')
            )
            return response, "NEED_CONTEXT"
    
    # ========== 이후 모든 로직은 확장된 user_input 사용 ==========
    
    # 1. 욕설 차단
    user_clean = user_input.lower().replace(' ', '')
    BLOCKED_KEYWORDS = ['시발', '씨발', 'ㅅㅂ', '병신', 'ㅂㅅ', '지랄', 'ㅈㄹ', '개새끼', '꺼져', '닥쳐', '죽어', '미친', '존나', 'fuck']
    if any(kw in user_clean for kw in BLOCKED_KEYWORDS):
        response, response_type = handle_blocked(user_input, {}, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'blocked', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # 2. 인사말 처리
    greeting_keywords = ['안녕', '하이', '헬로', 'hello', 'hi', '반가워']
    if any(kw in user_clean for kw in greeting_keywords) and len(user_clean) < 15:
        response, response_type = handle_greeting(user_input, {}, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'greeting', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # 3. 컨텍스트 업데이트 (확장된 질문에서 제도/전공 추출)
    program_type = extract_program_from_text(user_input)
    entity_name, entity_type = extract_entity_from_text(user_input)
    
    if program_type:
        update_context_in_session(program=program_type)
    if entity_name:
        update_context_in_session(entity=entity_name, entity_type=entity_type)
    
    # 4. 프로그램명만 입력된 경우 → PROGRAM_INFO FAQ로 연결
    program_only_match = check_program_name_only(user_input)
    if program_only_match:
        debug_print(f"[DEBUG] 프로그램명만 입력됨: {program_only_match}")
        # FAQ에서 해당 프로그램의 PROGRAM_INFO 찾기
        program_info_faq = faq_df[
            (faq_df['program'] == program_only_match) & 
            (faq_df['intent'] == 'PROGRAM_INFO')
        ]
        if not program_info_faq.empty:
            faq_match = program_info_faq.iloc[0]
            raw_answer = faq_match.get('answer', '')
            program = faq_match.get('program', '')
            
            conversational_answer = generate_conversational_response(raw_answer, user_input, program)
            formatted_response = format_faq_response_html(conversational_answer, program)
            formatted_response += create_contact_box()
            
            # 컨텍스트 업데이트
            update_context_in_session(program=program_only_match)
            
            log_to_sheets(
                st.session_state.get('session_id', 'unknown'),
                original_input, formatted_response, 'faq_program_info', 
                time.time() - start_time,
                st.session_state.get('page', 'AI챗봇 상담')
            )
            return formatted_response, "FAQ_PROGRAM_INFO"
    
    # 4.5 프로그램 설명 질문 패턴 직접 처리 (예: "복수전공은 뭐야?", "부전공 설명해줘")
    # FAQ 키워드 미등록으로 매칭 실패하는 경우를 코드로 보완
    if program_type and not entity_name:
        _info_words = ['뭐야', '뭔지', '무엇', '설명', '알려줘', '뭐임', '뭐에요', '뭐죠', '어떤거', '어떤것', '어떤제도', '개념', '정의']
        _comparison_words = ['차이', '비교', '다른점', '다른거', 'vs', '차이점', '비교해']
        _user_clean_tmp = user_input.lower().replace(' ', '')
        _is_comparison = any(w in _user_clean_tmp for w in _comparison_words)
        if any(w in _user_clean_tmp for w in _info_words) and not _is_comparison:
            _prog_display = MAPPINGS.get('program_display_names', {}).get(program_type, program_type)
            _pi_faq = faq_df[
                (faq_df['program'].isin([program_type, _prog_display])) &
                (faq_df['intent'] == 'PROGRAM_INFO')
            ]
            if not _pi_faq.empty:
                faq_match = _pi_faq.iloc[0]
                raw_answer = faq_match.get('answer', '')
                program = faq_match.get('program', '')
                conversational_answer = generate_conversational_response(raw_answer, user_input, program)
                formatted_response = format_faq_response_html(conversational_answer, program)
                formatted_response += create_contact_box()
                update_context_in_session(program=program_type)
                log_to_sheets(
                    st.session_state.get('session_id', 'unknown'),
                    original_input, formatted_response, 'faq_program_info_direct',
                    time.time() - start_time,
                    st.session_state.get('page', 'AI챗봇 상담')
                )
                return formatted_response, "FAQ_PROGRAM_INFO"

    # 5. FAQ 매핑 검색 (확장된 질문으로!)
    faq_match, score = search_faq_mapping(user_input, faq_df)
    
    if faq_match is not None and score >= 10:
        raw_answer = faq_match.get('answer', '')
        program = faq_match.get('program', '')
        
        conversational_answer = generate_conversational_response(raw_answer, user_input, program)
        formatted_response = format_faq_response_html(conversational_answer, program)
        formatted_response += create_contact_box()
        
        response_type = f"FAQ_{faq_match.get('intent', 'UNKNOWN')}"
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, formatted_response, 'faq', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return formatted_response, response_type
    
    # 5. 특수 핸들러 (연락처, 과목 검색, 전공 정보 등)
    # 간단한 의도 분류
    extracted_info = {
        'entity': entity_name,
        'entity_type': entity_type,
        'program': program_type,
        'major': entity_name
    }
    
    # 연락처 문의
    contact_keywords = ['연락처', '전화번호', '번호', '문의처', '사무실', '팩스']
    if any(kw in user_clean for kw in contact_keywords):
        response, response_type = handle_contact_search(user_input, extracted_info, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'contact', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # 교과목 검색
    course_keywords = ['교과목', '과목', '커리큘럼', '수업', '강의', '이수체계도', '교육과정', '뭐들어', '뭐배워']
    if entity_name and any(kw in user_clean for kw in course_keywords):
        response, response_type = handle_course_search(user_input, extracted_info, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'course', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # 전공 목록 검색
    list_keywords = ['목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '뭐가있어', '뭐있어']
    if program_type and any(kw in user_clean for kw in list_keywords):
        response, response_type = handle_major_search(user_input, extracted_info, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'major_list', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # 특정 전공 정보
    if entity_name and not any(kw in user_clean for kw in contact_keywords + course_keywords):
        response, response_type = handle_major_info(user_input, extracted_info, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'major_info', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, response_type
    
    # ========== 🔧 수정: AI Fallback 전 검증 ==========
    # 제도명이나 전공명이 명확하지 않으면 AI에게 넘기지 않고 재질문 유도
    
    user_clean_check = user_input.lower().replace(' ', '')
    
    # 제도 키워드 체크
    program_keywords = ['복수전공', '부전공', '융합전공', '융합부전공', '마이크로', '마이크로디그리', '소단위', '소단위전공', '소단위전공과정', '연계전공', 'md', '다전공', '유연학사제도', '유연학사', '유연제도']
    has_program_keyword = any(kw in user_clean_check for kw in program_keywords)
    
    # 전공명 체크 (실제 전공 데이터에서)
    has_specific_major = False
    if not MAJORS_INFO.empty and '전공명' in MAJORS_INFO.columns:
        for _, row in MAJORS_INFO.iterrows():
            major_name = str(row.get('전공명', '')).replace(' ', '').lower()
            if len(major_name) > 2 and major_name in user_clean_check:
                has_specific_major = True
                break
    
    # 마이크로디그리 과정명 체크
    if not has_specific_major and not MICRODEGREE_INFO.empty and '과정명' in MICRODEGREE_INFO.columns:
        for _, row in MICRODEGREE_INFO.iterrows():
            course_name = str(row.get('과정명', '')).replace(' ', '').lower()
            if len(course_name) > 2 and course_name in user_clean_check:
                has_specific_major = True
                break
    
    # 🔧 제도명도 전공명도 없으면 → 재질문 유도 (AI에게 넘기지 않음)
    if not has_program_keyword and not has_specific_major:
        debug_print("[DEBUG] ⚠️ 제도/전공 키워드 없음 - 재질문 유도")
        response = create_header_card("질문을 조금 더 구체적으로 해주세요", "🤔", "#f39c12")
        response += """
<div style="background: white; border-radius: 12px; padding: 16px; margin: 12px 0; border-left: 4px solid #f39c12;">
    <p style="margin: 0 0 12px 0; color: #333;">
        정확한 답변을 드리기 위해 <strong>제도명</strong>이나 <strong>전공명</strong>을 함께 말씀해 주세요!
    </p>
    <div style="background: #fff9e6; border-radius: 8px; padding: 12px; margin-top: 8px;">
        <p style="margin: 0 0 8px 0; color: #666; font-size: 0.9rem;"><strong>📌 이렇게 질문해 보세요:</strong></p>
        <ul style="margin: 0; padding-left: 20px; color: #555; font-size: 0.85rem;">
            <li>"<strong>복수전공</strong> 신청 기간이 언제야?"</li>
            <li>"<strong>융합전공</strong>이랑 <strong>복수전공</strong> 차이가 뭐야?"</li>
            <li>"<strong>경영학전공</strong> 교과목 알려줘"</li>
            <li>"<strong>마이크로디그리</strong> 목록 보여줘"</li>
        </ul>
    </div>
</div>
"""
        response += create_contact_box()
        
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'need_clarification', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return response, "NEED_CLARIFICATION"
    
    # 6. AI Fallback - 제도/전공이 명확한 경우에만 실행
    try:
        # 관련 FAQ 찾기
        related_faqs = []
    
        for _, row in faq_df.iterrows():
            program = str(row.get('program', '')).replace(' ', '')
            keywords = str(row.get('keyword', '')).split(',')
        
            # 프로그램명이나 키워드가 질문에 포함되면
            if program in user_clean:
                keyword_match = any(kw.strip().replace(' ', '').lower() in user_clean 
                                for kw in keywords if kw.strip())
                if keyword_match:
                    related_faqs.append(row)
    
        # FAQ 컨텍스트 생성
        faq_context = ""
        if related_faqs:
            faq_context = "\n\n[참고 FAQ 정보]\n"
            for faq in related_faqs[:3]:
                faq_context += f"""
    **{faq.get('program', '')} - {faq.get('intent', '')}**
    {faq.get('answer', '')}
    ---
    """
    
        # 프로그램 정보 컨텍스트 생성
        context_parts = []
        programs = data_dict.get('programs', {})

        if programs:
            for prog_name, prog_info in programs.items():
                context_parts.append(f"[{prog_name}]\n- 설명: {prog_info.get('description', '')}\n- 이수학점: {prog_info.get('credits_multi', '')}\n- 신청자격: {prog_info.get('qualification', '')}")
    
        context = "\n\n".join(context_parts[:5])
    
        # AI 프롬프트
        prompt = f"""당신은 한경국립대학교 다전공 안내 AI챗봇입니다.

[중요 지침]
- 반드시 아래 제공된 정보 내에서만 답변하세요
- 제공된 정보에 없는 내용은 "정확한 정보는 학사지원팀(031-670-5035)에 문의해주세요"라고 안내하세요
- 추측하거나 만들어내지 마세요
- URL 작성 시 마크다운 서식(**, __, *, _)으로 감싸지 마세요. URL은 그대로 작성하세요.

[프로그램 정보]
{context}

{faq_context}

[현재 학생 질문]
{user_input}

[답변 지침]
1. 위 정보에 답이 있으면 그 내용을 기반으로 답변
2. 위 정보에 답이 없으면 학사지원팀 문의 안내
3. "~합니다" 등 정중한 종결어미 사용
4. 핵심 정보를 간결하게 전달
5. 이모지 적절히 사용 (📅, 📋, ✅ 등)
6. 학사공지 확인 안내: {ACADEMIC_NOTICE_URL}
7. URL은 마크다운 볼드(**나 __)로 감싸지 말고 그대로 작성
"""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={'temperature': 0.3, 'max_output_tokens': 1000}  # 🔧 temperature 낮춤
        )
        
        ai_response = response.text.strip()
        
        # 🔧 수정: 더 엄격한 답변 검증
        failure_keywords = ['잘 모르겠', '확인할 수 없', '정보가 없습니다', '알 수 없', '찾을 수 없']
        is_failed = len(ai_response) < 10 or any(kw in ai_response for kw in failure_keywords)
        
        # 🔧 추가: AI가 엉뚱한 전공명을 생성했는지 검증
        # 응답에 질문에 없는 전공명이 있으면 실패로 처리
        suspicious_patterns = ['기계공학', '전자공학', '컴퓨터공학', '화학공학', '토목공학']
        for pattern in suspicious_patterns:
            if pattern in ai_response and pattern not in user_input:
                debug_print(f"[DEBUG] ⚠️ AI가 엉뚱한 전공 '{pattern}' 생성 - 차단")
                is_failed = True
                break
        
        if is_failed:
            log_failed_to_sheets(
                st.session_state.get('session_id', 'unknown'),
                original_input, ai_response, "AI가 적절한 답변을 생성하지 못함"
            )
            # 실패 시 재질문 유도
            response, response_type = handle_out_of_scope(user_input, extracted_info, data_dict)
            log_to_sheets(
                st.session_state.get('session_id', 'unknown'),
                original_input, response, 'ai_failed', 
                time.time() - start_time,
                st.session_state.get('page', 'AI챗봇 상담')
            )
            return response, response_type
        
        formatted_response = f"""
<div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea; border-radius: 12px; padding: 16px; margin: 12px 0;">
    {ai_response}
</div>
"""
        formatted_response += create_contact_box()
        
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, formatted_response, 'ai', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        return formatted_response, "AI_RESPONSE"
        
    except Exception as e:
        response, response_type = handle_out_of_scope(user_input, extracted_info, data_dict)
        log_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, response, 'failed', 
            time.time() - start_time,
            st.session_state.get('page', 'AI챗봇 상담')
        )
        log_failed_to_sheets(
            st.session_state.get('session_id', 'unknown'),
            original_input, str(e), "예외 발생"
        )
        return response, response_type


# ============================================================
# 📊 이수체계도 및 과목 표시 함수
# ============================================================

def display_curriculum_image(major, program_type):
    """이수체계도/과정 안내 이미지 표시"""
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
    if major.endswith(')') and '(' in major:
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            clean_major = major[:last_open_paren].strip()
    
    search_keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace('(', '').replace(')', '').replace(' ', '').strip()
    
    type_matched = CURRICULUM_MAPPING[CURRICULUM_MAPPING['제도유형'].apply(match_program_type_for_image)]
    
    if type_matched.empty:
        return
    
    filtered = type_matched[type_matched['전공명'] == clean_major]
    
    if filtered.empty:
        filtered = type_matched[type_matched['전공명'] == major]
    
    if filtered.empty:
        clean_major_no_space = clean_major.replace(' ', '')
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            cm_major_no_space = cm_major.replace(' ', '')
            if clean_major_no_space == cm_major_no_space:
                filtered = type_matched[type_matched['전공명'] == cm_major]
                break
    
    if filtered.empty and len(search_keyword) >= 2:
        for _, row in type_matched.iterrows():
            cm_major = str(row['전공명'])
            cm_keyword = cm_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace('(', '').replace(')', '').replace(' ', '').strip()
            if len(cm_keyword) >= 2 and len(search_keyword) >= 2:
                if search_keyword in cm_keyword or cm_keyword in search_keyword:
                    filtered = type_matched[type_matched['전공명'] == cm_major]
                    break
    
    if not filtered.empty:
        images_shown = 0
        missing_files = []
        total_images = len(filtered)
        
        for idx, row in filtered.iterrows():
            filename = row['파일명']
            
            if pd.notna(filename) and str(filename).strip():
                filename_str = str(filename).strip()
                
                if ',' in filename_str:
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
                        missing_files.append(filename_str)
        
        if missing_files:
            st.warning(f"⚠️ 다음 이미지 파일을 찾을 수 없습니다:")
            for missing_file in missing_files:
                st.caption(f"   • `{CURRICULUM_IMAGES_PATH}/{missing_file}`")
        
        if images_shown == 0 and not missing_files:
            st.caption("📷 이미지 파일 준비 중입니다.")
    else:
        st.info(f"💡 '{major}' 또는 '{clean_major}'에 해당하는 이미지 정보를 찾을 수 없습니다.")


def render_course_list(df, is_micro):
    for idx, row in df.iterrows():
        course_name = row.get('과목명', '')
        credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
        desc = row.get('교과목개요')

        title = f"📘 {course_name} ({credit})"

        with st.expander(title):
            if desc and pd.notna(desc) and str(desc).strip():
                st.write(desc)
            else:
                st.info("교과목 개요 정보가 없습니다.")

            edu_dept = row.get('교육 운영전공') or row.get('교육운영전공', '')
            if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                st.caption(f"🏫 운영전공: {str(edu_dept).strip()}")


def display_courses(major, program_type):
    """과목 정보 표시 - 수정 버전"""
    if not major or major == "선택 안 함":
        return False
    
    if COURSES_DATA.empty:
        st.info("교과목 데이터가 없습니다.")
        return False
    
    is_micro = "소단위" in program_type or "마이크로" in program_type
    
    def match_program_type_for_courses(type_value):
        """제도유형 매칭 - 개선 버전"""
        type_str = str(type_value).strip()
        type_list = [t.strip() for t in type_str.split(',')]
        
        if is_micro:
            return any(kw in type_str.lower() for kw in ['소단위', '마이크로', 'md'])
        
        if program_type == "부전공":
            return "부전공" in type_list and "융합부전공" not in type_list
        
        if program_type == "융합전공":
            return "융합전공" in type_list
        
        if program_type == "융합부전공":
            return "융합부전공" in type_list
        
        if program_type == "연계전공":
            return "연계전공" in type_list
        
        return program_type in type_list
    
    clean_major = major
    display_major = major
    
    if major.endswith(')') and '(' in major:
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            clean_major = major[:last_open_paren].strip()
            display_major = clean_major
    
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == clean_major) & 
        (COURSES_DATA['제도유형'].apply(match_program_type_for_courses))
    ]
    
    if courses.empty and is_micro:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '').replace('MD', '').replace(' ', '').strip()
        type_matched = COURSES_DATA[COURSES_DATA['제도유형'].apply(match_program_type_for_courses)]
        
        for course_major in type_matched['전공명'].unique():
            cm_str = str(course_major)
            if 'MD' in cm_str or 'md' in cm_str.lower():
                cm_keyword = cm_str.replace('MD', '').replace('md', '').replace(' ', '').strip()
                if len(keyword) >= 2 and len(cm_keyword) >= 2:
                    if keyword[:2] in cm_keyword or cm_keyword[:2] in keyword:
                        courses = type_matched[type_matched['전공명'] == course_major]
                        display_major = cm_str
                        break

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
        st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">📚 교과목 안내</p>', unsafe_allow_html=True)
        
        years = sorted([int(y) for y in courses['학년'].unique() if pd.notna(y)])
        
        if years:
            tabs = st.tabs([f"{year}학년" for year in years])
            
            for idx, year in enumerate(years):
                with tabs[idx]:
                    year_courses = courses[courses['학년'] == year]
                    semesters = sorted([int(s) for s in year_courses['학기'].unique() if pd.notna(s)])
                    
                    for semester in semesters:
                        st.markdown(f'<p style="font-size: 1.1rem; font-weight: 600; margin: 16px 0 12px 0;">📅 {semester}학기</p>', unsafe_allow_html=True)
                        semester_courses = year_courses[year_courses['학기'] == semester]
                        
                        required = semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)]
                        elective = semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if not required.empty:
                                st.markdown("**🔴 전공필수**")
                                render_course_list(required, is_micro)
                                                                    
                        with col2:
                            if not elective.empty:
                                st.markdown("**🟢 전공선택**")
                                render_course_list(elective, is_micro)
                        
                        st.divider()
        else:
            semesters = sorted([int(s) for s in courses['학기'].unique() if pd.notna(s)])
            
            if semesters:
                for semester in semesters:
                    st.markdown(f'<p style="font-size: 1.1rem; font-weight: 600; margin: 16px 0 12px 0;">📅 {semester}학기</p>', unsafe_allow_html=True)
                    semester_courses = courses[courses['학기'] == semester]
                    
                    has_required = not semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)].empty
                    has_elective = not semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)].empty
                    
                    if has_required or has_elective:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            required = semester_courses[semester_courses['이수구분'].str.contains('필수', na=False)]
                            if not required.empty:
                                st.markdown("**🔴 전공필수**")
                                render_course_list(required, is_micro)
                        
                        with col2:
                            elective = semester_courses[semester_courses['이수구분'].str.contains('선택', na=False)]
                            if not elective.empty:
                                st.markdown("**🟢 전공선택**")
                                render_course_list(elective, is_micro)
                    
                    # 🔥 수정: else 블록에서도 render_course_list 사용!
                    else:
                        st.markdown("**📚 교과목 목록**")
                        render_course_list(semester_courses, is_micro)
                    
                    st.divider()
            else:
                # 🔥 수정: 여기서도 render_course_list 사용!
                st.markdown("**📚 교과목 목록**")
                render_course_list(courses, is_micro)
        
        st.markdown("---")
        display_major_contact(display_major, program_type)
        return True
    else:
        st.info(f"'{display_major}' 교과목 정보가 없습니다.")
        return False

def display_major_contact(major, program_type="전공"):
    """전공 연락처 표시 - 마이크로디그리 지원"""
    
    # 🔥 마이크로디그리 체크
    is_micro = "소단위" in program_type or "마이크로" in program_type
    
    # 마이크로디그리인 경우 MICRODEGREE_INFO에서 찾기
    if is_micro and not MICRODEGREE_INFO.empty:
        clean_major = major
        
        # 괄호 제거
        if major.endswith(')') and '(' in major:
            last_open_paren = major.rfind('(')
            if last_open_paren > 0:
                clean_major = major[:last_open_paren].strip()
        
        # MD 제거
        clean_major = clean_major.replace(' MD', '').replace('MD', '').strip()
        
        # 🔥 MICRODEGREE_INFO에서 검색
        contact_row = pd.DataFrame()
        
        # 1차: 정확한 과정명 매칭
        if '과정명' in MICRODEGREE_INFO.columns:
            contact_row = MICRODEGREE_INFO[MICRODEGREE_INFO['과정명'] == major]
        
        # 2차: 괄호 제거 후 매칭
        if contact_row.empty:
            contact_row = MICRODEGREE_INFO[MICRODEGREE_INFO['과정명'] == clean_major]
        
        # 3차: 부분 매칭
        if contact_row.empty:
            keyword = clean_major.replace('전공', '').replace('과정', '').replace('전문가', '')
            if keyword:
                contact_row = MICRODEGREE_INFO[
                    MICRODEGREE_INFO['과정명'].str.contains(keyword, na=False, regex=False)
                ]
        
        # 마이크로디그리 정보 표시
        if not contact_row.empty:
            row = contact_row.iloc[0]
            
            course_name = row.get('과정명', major)
            edu_major = row.get('교육운영전공', '')
            phone = row.get('연락처', '')
            location = row.get('위치', row.get('사무실위치', ''))
            
            contact_parts = [f"🎓 **과정명**: {course_name}"]
            
            if pd.notna(edu_major) and str(edu_major).strip():
                contact_parts.append(f"🏛️ **교육운영전공**: {edu_major}")
            
            if pd.notna(phone) and str(phone).strip():
                contact_parts.append(f"📞 **연락처**: {phone}")
            
            if pd.notna(location) and str(location).strip():
                contact_parts.append(f"📍 **사무실 위치**: {location}")
            
            st.info(f"**📋 소단위전공과정 문의처**\n\n" + "\n\n".join(contact_parts))
            return
    
    # 일반 전공인 경우 MAJORS_INFO에서 찾기
    if MAJORS_INFO.empty:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")
        return
    
    edu_major = None
    clean_major = major
    if major.endswith(')') and '(' in major:
        last_open_paren = major.rfind('(')
        if last_open_paren > 0:
            edu_major = major[last_open_paren+1:-1].strip()
            clean_major = major[:last_open_paren].strip()
    
    clean_major = clean_major.replace(' MD', '').replace('MD', '').strip()
    
    contact_row = pd.DataFrame()
    if edu_major:
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == edu_major]

    if contact_row.empty:
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == clean_major]
    
    if contact_row.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('(', '').replace(')', '')[:4]
        if keyword:
            contact_row = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains(keyword, na=False, regex=False)]
    
    if not contact_row.empty:
        row = contact_row.iloc[0]
        major_name = row.get('전공명', major)
        phone = row.get('연락처', '')
        location = row.get('사무실위치', row.get('위치', ''))
        
        contact_title = f"{program_type} 문의처"
        
        contact_parts = [f"🎓 **전공명**: {major_name}"]
        if pd.notna(phone) and str(phone).strip():
            contact_parts.append(f"📞 **연락처**: {phone}")
        if pd.notna(location) and str(location).strip():
            contact_parts.append(f"📍 **사무실 위치**: {location}")
        
        st.info(f"**📋 {contact_title}**\n\n" + "\n\n".join(contact_parts))
    else:
        st.info(f"📞 **문의**: 학사지원팀 031-670-5035")


def render_question_buttons(questions, key_prefix, cols=5):
    btn_cols = st.columns(cols)
    for i, q in enumerate(questions):
        if btn_cols[i % cols].button(q, key=f"{key_prefix}_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response_text, res_type = generate_ai_response(q, st.session_state.chat_history[:-1], ALL_DATA)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
            st.session_state.should_scroll = True  # 스크롤 플래그 설정
            st.rerun()


# ============================================================
# 🖥️ 메인 UI
# ============================================================

def main():
    initialize_session_state()
    
    # 다크모드 대응 CSS
    st.markdown("""
    <style>
    /* 다크모드 감지 및 메뉴 색상 조정 */
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] .nav-link {
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] h3 {
            color: #fafafa !important;
        }
    }
    
    /* 라이트모드 */
    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] .nav-link {
            color: #262730 !important;
        }
        [data-testid="stSidebar"] h3 {
            color: #262730 !important;
        }
    }
    
    /* Streamlit 다크모드 클래스 대응 */
    [data-theme="dark"] .nav-link {
        color: #fafafa !important;
    }
    [data-theme="dark"] h3 {
        color: #fafafa !important;
    }
    
    [data-theme="light"] .nav-link {
        color: #262730 !important;
    }
    [data-theme="light"] h3 {
        color: #262730 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <p style='font-size: 3rem; margin-bottom: 0;'>🎓</p>
            <p style='margin-top: 0; font-size: 1.3rem; font-weight: 600;'>HKNU 다전공</p>
        </div>
        """, unsafe_allow_html=True)
        
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내", "다전공 비교 분석"], 
            icons=["chat-dots-fill", "journal-bookmark-fill", "calculator-fill"],
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "15px", 
                    "text-align": "left", 
                    "margin": "0px"
                },
                "nav-link-selected": {
                    "background-color": "#0091FF",
                    "color": "white"
                },
            }
        )
        
        st.divider()
        
        # AI챗봇 소개
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 4px solid #667eea; 
                    padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <p style="color: #333; margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 600;">
                🤖 챗봇 소개
            </p>
            <p style="color: #555; font-size: 0.82rem; margin: 0 0 8px 0; line-height: 1.6;">
                한경국립대 다전공 제도에 관한<br>
                궁금한 사항을 AI기반 챗봇이<br>
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
            <p style="color: #333; margin: 0 0 10px 0; font-size: 0.95rem; font-weight: 600;">
                📚 다전공 제도란?
            </p>
            <p style="color: #555; font-size: 0.82rem; margin: 0; line-height: 1.6;">
                주전공 외에 복수, 융합전공 등<br>
                다양한 학위를 취득하여<br>
                융합형 인재로 성장할 수 있도록<br>
                지원하는 유연학사제도입니다.
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
        
        # Powered by 정보
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
        st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">💬 챗봇과 대화하기</p>', unsafe_allow_html=True)

        with st.expander("💡 어떤 질문을 해야 할지 모르겠나요? **(클릭)**", expanded=False):

            # 질문 버튼 탭
            tab_apply, tab_program, tab_credit, tab_etc = st.tabs(
                ["📋 신청", "📚 제도", "🎓 학점", "🎯 전공/ 📞 연락처"]
            )
            
            with tab_apply:
                q_apply = [
                    "다전공 신청자격은?",
                    "복수전공 신청 기간은?",
                    "융합전공 신청 방법은 뭐야?",
                    "다전공을 변경하려면?",
                ]
                render_question_buttons(q_apply, "qa", cols=2)

            with tab_program:
                q_program = [
                    "다전공 제도가 뭐야?",
                    "복수전공은 뭐야?",
                    "마이크로디그리는 어떤 과정이 있어?",
                    "복수·부전공 차이는 뭐야?",
                ]
                render_question_buttons(q_program, "qp", cols=2)

            with tab_credit:
                q_credit = [
                    "다전공별 이수학점은?",
                    "복수전공 이수학점 알려줘",
                    "융합전공의 졸업학점은?",
                    "마이크로디그리 과정의 이수학점은?",
                ]
                render_question_buttons(q_credit, "qc", cols=2)

            with tab_etc:
                q_etc = [
                    "경영학전공 연락처 알려줘",
                    "응용수학전공 사무실은 어디야?",
                    "기계공학전공 교과목은?",
                    "AI빅데이터융합전공 교과목 알려줘",
                ]
                render_question_buttons(q_etc, "qe", cols=2)

            st.divider()
        
        # 채팅 히스토리 표시
        for chat in st.session_state.chat_history:
            avatar = "🧑‍🎓" if chat["role"] == "user" else "🤖"
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"], unsafe_allow_html=True)
        
        # 스크롤 플래그 확인 및 실행
        if st.session_state.should_scroll:
            scroll_to_bottom()
            st.session_state.should_scroll = False  # 플래그 리셋
        
        # 채팅 입력
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
        <p style="font-size: 2rem; margin-bottom: 20px; color: #1f2937; font-weight: 600;">
            📊 제도 한눈에 비교
        </p>
        """, unsafe_allow_html=True)
        
        # 제도 비교 카드
        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '')[:50] + '...' if len(info.get('description', '')) > 50 else info.get('description', '-')
                    qual = info.get('qualification', '-')[:30] + '...' if len(str(info.get('qualification', '-'))) > 30 else info.get('qualification', '-')
                    
                    html = f"""<div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-height: 400px; margin-bottom: 12px;"><p style="margin: 0 0 8px 0; color: #1f2937; font-size: 1rem; font-weight: 600;">🎓 {program}</p><p style="color: #6b7280; font-size: 11px; margin-bottom: 10px; line-height: 1.4;">{desc}</p><hr style="margin: 8px 0; border-top: 1px solid #e5e7eb;"><div style="font-size: 12px; margin-bottom: 8px;"><strong>📖 이수학점</strong><br><span style="font-size: 11px; line-height: 1.6;">• 본전공: {info.get('credits_primary', '-')}<br>• 다전공: {info.get('credits_multi', '-')}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>✅ 신청자격</strong><br><span style="font-size: 11px; color: #4b5563;">{qual}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>🎓 졸업요건</strong><br><span style="font-size: 11px;">졸업인증: {info.get('graduation_certification', '-')}<br>졸업시험: {info.get('graduation_exam', '-')}</span></div><div style="font-size: 12px; margin-bottom: 6px;"><strong>📜 학위표기</strong><br><span style="font-size: 11px; color: #2563eb;">{str(info.get('degree', '-'))[:30]}</span></div><div style="text-align: right; margin-top: 10px;"><span style="font-size: 11px;">난이도: </span><span style="color: #f59e0b;">{info.get('difficulty', '⭐⭐⭐')}</span></div></div>"""
                    st.markdown(html, unsafe_allow_html=True)
        
        st.divider()
        st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">🔍 상세 정보 조회</p>', unsafe_allow_html=True)
        
        prog_keys = list(ALL_DATA['programs'].keys()) if 'programs' in ALL_DATA else []
        selected_program = st.selectbox("제도 선택", prog_keys)
        
        if selected_program:
            info = ALL_DATA['programs'][selected_program]
            
            tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징"])
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**개요**\n\n{info.get('description', '-')}")
                    
                    credits_text = f"""**이수학점**
- 교양: {info.get('credits_general', '-')}
- 원전공: {info.get('credits_primary', '-')}
- 다전공: {info.get('credits_multi', '-')}"""
                    st.markdown(credits_text)
                    
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
                    return "융합전공" in type_str
                return selected_prog in type_str
            
            if not COURSES_DATA.empty and '제도유형' in COURSES_DATA.columns:
                mask = COURSES_DATA['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                for major in COURSES_DATA[mask]['전공명'].unique():
                    available_majors[major] = None
            
            if not MAJORS_INFO.empty and '제도유형' in MAJORS_INFO.columns:
                mask = MAJORS_INFO['제도유형'].apply(lambda x: match_program_type(x, selected_program))
                for _, row in MAJORS_INFO[mask].iterrows():
                    if selected_program == "융합부전공":
                        continue
                    major_name = row['전공명']
                    edu_major = row.get('교육운영전공')
                    if pd.notna(edu_major) and str(edu_major).strip():
                        available_majors[major_name] = str(edu_major).strip()
                    elif major_name not in available_majors:
                        available_majors[major_name] = None
            
            if available_majors:
                target_programs = ["복수전공", "부전공", "융합전공", "융합부전공", "연계전공"]
    
                # 🔥 구분 명확히
                is_microdegree = any(sp in selected_program for sp in ["소단위", "마이크로"])
                is_linked = "연계전공" in selected_program
                is_convergence = any(sp in selected_program for sp in ["융합전공", "융합부전공"])
    
                # [수정] 카테고리 설정 로직 변경
                category_majors = {}

                if is_microdegree or is_convergence:
                    # 융합전공, 마이크로는 '전체' 하나로 통일
                    category_majors = {"전체": sorted(available_majors.keys())}
                elif is_linked:
                    # 🔥 [핵심 수정] 연계전공을 '계열' 별로 분류하는 로직 추가
                    target_col = '계열' if '계열' in MAJORS_INFO.columns else ('단과대학' if '단과대학' in MAJORS_INFO.columns else None)
                
                    if target_col:
                        for major_name in available_majors.keys():
                            # MAJORS_INFO에서 해당 전공의 행을 찾음
                            major_row = MAJORS_INFO[MAJORS_INFO['전공명'] == major_name]
                        
                            if not major_row.empty:
                                # 해당 전공의 계열 정보를 가져옴 (여러 개일 경우 첫 번째 것 사용)
                                cat_val = major_row.iloc[0].get(target_col)
                                category = str(cat_val).strip() if pd.notna(cat_val) else "기타"
                            else:
                                category = "기타"
                        
                            if category not in category_majors:
                                category_majors[category] = []
                            category_majors[category].append(major_name)
                    
                        # 딕셔너리 키 정렬 (가나다순)
                        category_majors = dict(sorted(category_majors.items()))
                    else:
                        # 계열 컬럼을 못 찾으면 전체로 표시
                        category_majors = {"전체": sorted(available_majors.keys())}
                else:
                    category_majors = get_majors_by_category(selected_program)
    
                if selected_program in target_programs:
                    # 🔥 1. 연계전공: 단일 컬럼만
                    if is_linked:
                        major_options_with_dividers = ["선택 안 함"]

                        for category in sorted(category_majors.keys()):
                            divider = f"━━━━━━ {category} ━━━━━━"
                            major_options_with_dividers.append(divider)
                            for major in sorted(category_majors[category]):
                                major_options_with_dividers.append(major)

                        selected_major = st.selectbox(
                        f"🎓 이수하려는 {selected_program}",
                        major_options_with_dividers
                        )

                        # [수정 3] 구분선 선택 시 경고 및 null 처리
                        if selected_major and "━━━" in selected_major:
                            st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            selected_major = None
                            
                        my_primary = "선택 안 함"
                        admission_year = datetime.now().year
        
                    # 🔥 2. 융합전공: 전공 + 본전공 + 학번
                    elif is_convergence or len(category_majors) <= 1:
                        col_m1, col_m2, col_m3 = st.columns([3, 3, 1.5])
                        with col_m1:
                            all_majors = []
                            for majors in category_majors.values():
                                all_majors.extend(majors)
                            selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(set(all_majors)))
                        with col_m2:
                            primary_categories = get_majors_by_category("복수전공")
                            if len(primary_categories) > 1:
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
        
                    # 🔥 3. 복수전공/부전공: 일반 처리 (기존 코드)
                    else:
                        major_options_with_dividers = ["선택 안 함"]
                        major_to_category = {}
            
                        for category in sorted(category_majors.keys()):
                            divider = f"━━━━━━ {category} ━━━━━━"
                            major_options_with_dividers.append(divider)
                            for major in sorted(category_majors[category]):
                                major_options_with_dividers.append(major)
                                major_to_category[major] = category
            
                        primary_categories = get_majors_by_category("복수전공")
                        primary_options_with_dividers = ["선택 안 함"]
            
                        for category in sorted(primary_categories.keys()):
                            divider = f"━━━━━━ {category} ━━━━━━"
                            primary_options_with_dividers.append(divider)
                            for major in sorted(primary_categories[category]):
                                primary_options_with_dividers.append(major)
            
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
            
                        if selected_major and "━━━" in selected_major:
                            st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            selected_major = None
            
                        if my_primary and "━━━" in my_primary:
                            st.warning("⚠️ 계열 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            my_primary = "선택 안 함"
        
                else:
                    # 🔥 소단위전공과정(마이크로디그리) - MICRODEGREE_INFO 사용
                    field_majors = {}
                    major_to_edu_major = {}
                    
                    if not MICRODEGREE_INFO.empty and '과정명' in MICRODEGREE_INFO.columns:
                        group_column = '계열' if '계열' in MICRODEGREE_INFO.columns else None
                        
                        for _, row in MICRODEGREE_INFO.iterrows():
                            if group_column:
                                field = row.get(group_column, '기타')
                                if pd.isna(field) or str(field).strip() == '':
                                    field = '기타'
                                field = str(field).strip()
                            else:
                                field = '전체'
                            
                            course_name = row.get('과정명', '')
                            edu_major = row.get('교육운영전공', '')
                            
                            if pd.notna(edu_major) and str(edu_major).strip():
                                display_name = f"{course_name}({str(edu_major).strip()})"
                                major_to_edu_major[display_name] = str(edu_major).strip()
                            else:
                                display_name = course_name
                                major_to_edu_major[display_name] = course_name
                            
                            if field not in field_majors:
                                field_majors[field] = []
                            if display_name not in field_majors[field]:
                                field_majors[field].append(display_name)
                    
                    if field_majors and len(field_majors) > 1:
                        major_options_with_dividers = ["선택 안 함"]
                        
                        for field in sorted(field_majors.keys()):
                            divider = f"━━━━━━ {field} ━━━━━━"
                            major_options_with_dividers.append(divider)
                            for major in sorted(field_majors[field]):
                                major_options_with_dividers.append(major)
                        
                        selected_major = st.selectbox(
                            f"🎓 이수하려는 {selected_program}",
                            major_options_with_dividers,
                            key=f"micro_major_{selected_program}"
                        )
                        
                        if selected_major and "━━━" in selected_major:
                            st.warning("⚠️ 분야 구분선이 아닌 구체적인 전공명을 선택해주세요.")
                            selected_major = None
                    elif field_majors:
                        all_majors = []
                        for majors in field_majors.values():
                            all_majors.extend(majors)
                        
                        selected_major = st.selectbox(
                            f"🎓 이수하려는 {selected_program}",
                            ["선택 안 함"] + sorted(all_majors),
                            key=f"micro_major_{selected_program}"
                        )
                    else:
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
                    admission_year = datetime.now().year
                
                if selected_major:
                    if selected_program in target_programs and "연계전공" not in selected_program:
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.markdown(f'<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">🎯 {selected_program} 이수학점</p>', unsafe_allow_html=True)
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
                                        st.markdown(f'<p style="font-size: 1.1rem; font-weight: 600; margin: 12px 0;">👉 합계 {int(row.get("다전공_계", 0))}학점</p>', unsafe_allow_html=True)
                        
                        with col_r:
                            st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">🏠 본전공 이수학점 변화(신입학 기준)</p>', unsafe_allow_html=True)
                            if my_primary != "선택 안 함" and not PRIMARY_REQ.empty:
                                # 신입학 기준으로 필터링
                                pri_data = PRIMARY_REQ[
                                    (PRIMARY_REQ['전공명'] == my_primary) & 
                                    (PRIMARY_REQ['입학구분'] == '신입학')
                                ].copy()
                                if not pri_data.empty:
                                    pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
                                    pri_valid = pri_data[pri_data['기준학번'] <= admission_year].sort_values('기준학번', ascending=False)
                                    
                                    found_req = False

                                    for _, p_row in pri_valid.iterrows():
                                        if selected_program in str(p_row['제도유형']):
                                            # ✅ [수정 핵심] NaN(빈값) 처리를 위한 안전한 변환 로직
                                            def safe_int(val):
                                                try:
                                                    # 값이 없거나 NaN이면 0 반환
                                                    if pd.isna(val) or str(val).strip() == "":
                                                        return 0
                                                    # 실수형(3.0)도 정수(3)로 변환
                                                    return int(float(val))
                                                except:
                                                    return 0

                                            p_req = safe_int(p_row.get('본전공변화_전공필수'))
                                            p_sel = safe_int(p_row.get('본전공변화_전공선택'))
                                            p_total = safe_int(p_row.get('본전공변화_계'))

                                            st.write(f"전공필수: **{p_req}**학점")
                                            st.write(f"전공선택: **{p_sel}**학점")
                                            st.markdown(f'<p style="font-size: 1.1rem; font-weight: 600; margin: 12px 0;">👉 합계 {p_total}학점</p>', unsafe_allow_html=True)
                                            
                                            # 선택한 학번과 적용된 기준학번이 다르면 안내 문구 표시
                                            applied_year = int(p_row.get('기준학번', 0))
                                            if admission_year != applied_year:
                                                st.info(f"ℹ️ {applied_year}학번 기준 ({admission_year}학번 기준은 추후 업데이트 예정)")
                                            
                                            found_req = True
                                            break
                                    
                                    if not found_req:
                                        st.info("해당 학번/과정에 대한 본전공 요건 정보가 없습니다.")
                            else:
                                st.info("본전공을 선택하면 변동 학점을 확인할 수 있습니다.")
                    
                    st.divider()

                    if not MAJORS_INFO.empty and '전공설명' in MAJORS_INFO.columns:
                        # 선택된 전공에 해당하는 행 찾기
                        desc_row = MAJORS_INFO[MAJORS_INFO['전공명'] == selected_major]
                        
                        if not desc_row.empty:
                            # 전공설명 값 가져오기
                            description = desc_row.iloc[0].get('전공설명')
                            
                            # 내용이 비어있지 않다면(NaN이나 빈 문자열이 아니면) 출력
                            if pd.notna(description) and str(description).strip():
                                st.markdown(f'<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">📘 ({selected_program}) {selected_major} 전공 소개</p>', unsafe_allow_html=True)
                                st.info(str(description).strip())

                    if selected_program == "융합전공":
                        st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">📋 이수체계도</p>', unsafe_allow_html=True)
                        display_curriculum_image(selected_major, selected_program)
                        display_courses(selected_major, selected_program)
                    elif "소단위" in selected_program or "마이크로" in selected_program:
                        st.markdown('<p style="font-size: 1.3rem; font-weight: 600; margin: 20px 0 16px 0;">🖼️ 과정 안내 이미지</p>', unsafe_allow_html=True)
                        display_curriculum_image(selected_major, selected_program)
                        display_courses(selected_major, selected_program)
                    else:
                        display_courses(selected_major, selected_program)
            else:
                st.warning(f"⚠️ {selected_program}에 해당하는 전공 목록을 찾을 수 없습니다.")
                st.info("💡 데이터 파일에 해당 제도의 전공 정보가 있는지 확인해주세요.")

    # 🎯 다전공 비교 분석
    elif menu == "다전공 비교 분석":
        from simulation import render_simulation_page
        render_simulation_page()

if __name__ == "__main__":
    initialize_session_state()
    main()