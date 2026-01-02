"""
============================================================
🎓 한경국립대학교 다전공 안내 AI챗봇
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
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
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
        'Get help': 'https://www.hknu.ac.kr',
        'Report a bug': 'https://www.hknu.ac.kr',
        'About': "# 한경국립대학교 다전공 안내 AI 챗봇"
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
FAQ_MAPPING = load_faq_mapping()
MAJORS_INFO = load_majors_info()
GRADUATION_REQ = load_graduation_requirements()
PRIMARY_REQ = load_primary_requirements()

ALL_DATA = {
    'programs': PROGRAM_INFO,
    'curriculum': CURRICULUM_MAPPING,
    'courses': COURSES_DATA,
    'faq_mapping': FAQ_MAPPING,
    'majors': MAJORS_INFO,
    'grad_req': GRADUATION_REQ,
    'primary_req': PRIMARY_REQ,
}


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
    '마이크로디그리': ['마이크로디그리', '마이크로', 'md'],
}

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
        "학점이 몇 학점이야?", "이수 학점 알려줘", "졸업하려면 몇 학점 필요해?",
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
            api_key=st.secrets.get("GEMINI_API_KEY", "")
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
    
    # 우선순위: 더 긴 키워드 먼저 매칭
    program_order = ['소단위전공과정', '마이크로디그리', '융합부전공', '융합전공', '복수전공', '부전공', '연계전공']
    
    for program in program_order:
        keywords = PROGRAM_KEYWORDS.get(program, [program])
        for kw in keywords:
            if kw.lower().replace(' ', '') in text_lower:
                return program
    
    # '다전공' 일반 키워드
    if '다전공' in text_lower:
        return '다전공'
    
    return None


def search_faq_mapping(user_input, faq_df):
    """
    faq_mapping에서 가장 적합한 답변 검색
    - 제도(program)와 키워드가 모두 매칭될 때만 답변 반환
    - 그 외에는 None 반환하여 라우터로 넘김
    
    Returns: (faq_row, match_score) or (None, 0)
    """
    if faq_df.empty:
        return None, 0
    
    user_clean = user_input.lower().replace(' ', '')
    
    # 1. 프로그램(제도) 추출
    detected_program = extract_program_from_text(user_input)
    
    # 제도가 감지되지 않으면 FAQ 매칭 스킵 (라우터로 넘김)
    if not detected_program:
        return None, 0
    
    # 2. 해당 제도의 FAQ만 필터링
    if detected_program in ['소단위전공과정', '마이크로디그리']:
        # 소단위전공과정과 마이크로디그리는 동일 제도
        program_faq = faq_df[faq_df['program'].isin(['소단위전공과정', '마이크로디그리', '다전공'])]
    elif detected_program == '다전공':
        # '다전공' 일반 질문은 '다전공' 행만 검색
        program_faq = faq_df[faq_df['program'] == '다전공']
    else:
        # 특정 제도 + 일반 다전공 행도 포함
        program_faq = faq_df[faq_df['program'].isin([detected_program, '다전공'])]
    
    if program_faq.empty:
        return None, 0
    
    # 3. 키워드 매칭
    best_match = None
    best_score = 0
    
    for _, row in program_faq.iterrows():
        keywords = str(row.get('keyword', '')).split(',')
        keywords = [k.strip().lower().replace(' ', '') for k in keywords if k.strip()]
        
        # 키워드 매칭 개수
        keyword_matches = sum(1 for kw in keywords if kw and kw in user_clean)
        
        if keyword_matches == 0:
            continue  # 키워드 매칭 없으면 스킵
        
        score = keyword_matches * 10
        
        # 정확한 제도 매칭 보너스
        row_program = str(row.get('program', '')).strip()
        if row_program == detected_program:
            score += 30  # 정확한 제도 매칭
        elif row_program == '다전공':
            score += 10  # 일반 다전공 (낮은 우선순위)
        
        if score > best_score:
            best_score = score
            best_match = row
    
    # 최소 점수 기준: 키워드 1개 이상 매칭 필수 (score >= 10)
    if best_score >= 10:
        return best_match, best_score
    
    return None, 0


def generate_conversational_response(faq_answer, user_input, program=None):
    """FAQ 답변을 AI를 통해 대화체로 변환"""
    try:
        prompt = f"""당신은 한경국립대학교 다전공 안내 챗봇입니다. 
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
5. 문장 끝마다 줄바꿈 추가
6. 마지막에 추가 질문이 있는지 물어보세요
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
    return f"""<h3 style="margin: 20px 0 16px 0; font-size: 1.3rem; color: #333; font-weight: 600;">{emoji} {title}</h3>"""


def create_info_card(title, content_list, border_color="#007bff", emoji="📌"):
    items_html = ""
    for item in content_list:
        items_html += f'<p style="margin: 6px 0 6px 20px; font-size: 0.95rem; color: #333;">• {item}</p>\n'
    return f"""<div style="margin: 12px 0;"><h4 style="color: #333; margin: 10px 0 8px 0; font-size: 1rem; font-weight: 600;">{emoji} {title}</h4>{items_html}</div>"""


def create_simple_card(content, bg_color="#f0f7ff", border_color="#007bff"):
    return f"""<div style="margin: 12px 0; padding: 0;">{content}</div>"""


def create_step_card(step_num, title, description, color="#007bff"):
    return f"""<div style="display: flex; align-items: flex-start; margin: 12px 0; padding: 12px; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);"><div style="background: {color}; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 14px; flex-shrink: 0;">{step_num}</div><div><strong style="color: #333; font-size: 0.95rem;">{title}</strong><p style="margin: 4px 0 0 0; color: #666; font-size: 0.9rem;">{description}</p></div></div>"""


def create_tip_box(text, emoji="💡"):
    return f"""<p style="margin: 12px 0; color: #666; font-size: 0.9rem; font-style: italic;">{emoji} <strong>TIP:</strong> {text}</p>"""


def create_warning_box(text, emoji="⚠️"):
    return f"""<p style="margin: 12px 0; color: #dc3545; font-size: 0.9rem; font-weight: 500;">{emoji} {text}</p>"""


def create_contact_box():
    return """<p style="margin: 16px 0 0 0; color: #666; font-size: 0.9rem;">📞 <strong>문의:</strong> 전공 사무실 또는 학사지원팀 <strong>031-670-5035</strong></p>"""


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
    # URL 링크 변환
    url_pattern = r'(https?://[^\s]+)'
    answer = re.sub(url_pattern, r'<a href="\1" target="_blank" style="color: #007bff; text-decoration: underline;">\1</a>', answer)
    
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
            formatted_lines.append(f'<li style="margin: 5px 0; color: #333;">{content}</li>')
        else:
            if in_list:
                formatted_lines.append('</ol>')
                in_list = False
            formatted_lines.append(f'<p style="margin: 8px 0; color: #333; line-height: 1.6;">{line}</p>')
    
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
<div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%); border-left: 4px solid {color}; border-radius: 12px; padding: 16px; margin: 12px 0;">
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


def classify_intent(user_input, use_ai_fallback=True):
    """통합 의도 분류 함수"""
    user_clean = user_input.lower().replace(' ', '')
    
    # 1. 욕설 차단
    if any(kw in user_clean for kw in BLOCKED_KEYWORDS):
        return 'BLOCKED', 'blocked', {}
    
    # 2. 연락처/전화번호 문의 (최우선)
    contact_keywords = ['연락처', '전화번호', '번호', '문의처', '사무실', '팩스', 'contact', 'call']
    if any(kw in user_clean for kw in contact_keywords):
        return 'CONTACT_SEARCH', 'keyword', extract_additional_info(user_input, 'CONTACT_SEARCH')
    
    # 3. 정보 추출
    major_regex = r'([가-힣A-Za-z0-9]+(?:학과|전공|학부|교실|스쿨))'
    major_match = re.search(major_regex, user_clean)
    
    detected_major_name = None
    is_real_major = False
    
    if major_match:
        detected_major_name = major_match.group(1)
        system_keywords = ['복수전공', '부전공', '융합전공', '연계전공', '심화전공', '다전공', '마이크로전공', '전공']
        if detected_major_name not in system_keywords:
            is_real_major = True
    
    found_programs = extract_programs(user_clean)
    
    explicit_program_keywords = ['복수', '부전공', '다전공', '융합', '연계', '마이크로', '트랙', '심화']
    has_explicit_program = any(kw in user_clean for kw in explicit_program_keywords)
    
    if is_real_major and found_programs and not has_explicit_program:
        found_programs = []
    
    # 4. 교과목/커리큘럼 검색
    has_course_keyword = any(kw in user_clean for kw in ['교과목', '과목', '커리큘럼', '수업', '강의', '이수체계도'])
    if is_real_major and has_course_keyword:
        return 'COURSE_SEARCH', 'complex', {'major': detected_major_name}
    
    # 5. 복합 의도 (학과 + 프로그램)
    if is_real_major and found_programs:
        program = found_programs[0]
        if any(kw in user_clean for kw in ['신청', '지원', '하고싶', '원해', '어떻게', '방법']):
            return 'APPLY_METHOD', 'complex', {'program': program, 'major': detected_major_name}
        if any(kw in user_clean for kw in ['자격', '조건', '가능', '되나요']):
            return 'APPLY_QUALIFICATION', 'complex', {'program': program, 'major': detected_major_name}
        return 'PROGRAM_INFO', 'complex', {'program': program, 'major': detected_major_name}
    
    # 6. 학과 안내
    if is_real_major:
        return 'MAJOR_INFO', 'complex', {'major': detected_major_name}
    
    # 7. 프로그램 단독 문의
    if found_programs:
        program = found_programs[0]
        if any(kw in user_clean for kw in ['자격', '신청할수있', '조건', '대상', '기준']):
            return 'APPLY_QUALIFICATION', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['언제', '기간', '마감', '날짜', '일정', '시기']):
            return 'APPLY_PERIOD', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['어떻게', '방법', '절차', '순서', '경로']):
            return 'APPLY_METHOD', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['학점', '몇학점', '이수학점']):
            return 'CREDIT_INFO', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['등록금', '수강료', '학비', '장학금']):
            return 'PROGRAM_TUITION', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['취소', '포기', '철회', '그만']):
            return 'APPLY_CANCEL', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['변경', '바꾸', '전환']):
            return 'APPLY_CHANGE', 'complex', {'program': program}
        if any(kw in user_clean for kw in ['차이', '비교', 'vs']):
            return 'PROGRAM_COMPARISON', 'complex', {'program': program}
        return 'PROGRAM_INFO', 'inferred', {'program': program}
    
    # 8. Semantic Router
    if SEMANTIC_ROUTER is not None:
        semantic_intent, score = classify_with_semantic_router(user_input)
        if semantic_intent:
            return semantic_intent, 'semantic', extract_additional_info(user_input, semantic_intent)
    
    # 9. AI Fallback
    if use_ai_fallback:
        try:
            ai_intent = classify_with_ai(user_input)
            if ai_intent not in ['OUT_OF_SCOPE', 'BLOCKED']:
                return ai_intent, 'ai', extract_additional_info(user_input, ai_intent)
        except:
            pass
    
    return 'OUT_OF_SCOPE', 'fallback', {}


# ============================================================
# 🏫 계열별 전공 그룹화 헬퍼 함수
# ============================================================

def get_majors_by_category(program_type=None, data_source="majors"):
    """계열별로 전공을 그룹화하여 반환"""
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
        category_majors = get_majors_by_category()
        if category_majors and len(category_majors) > 1:
            response += "<div style='margin-top: 12px;'><strong>📚 계열별 전공 목록</strong></div>"
            response += format_majors_by_category_html(category_majors)
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
        response = create_header_card(f"'{major}' 과목 조회 실패", "📚", "#ff6b6b")
        response += create_warning_box(f"입력하신 <strong>'{major}'</strong> 전공을 찾을 수 없습니다.")
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


    response += f"""
<div style="background: white; border-left: 4px solid #11998e; border-radius: 8px; padding: 16px; margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <p style="margin: 8px 0; color: #333;"><strong>🎓 전공명:</strong> {row['전공명']}</p>
    <p style="margin: 8px 0; color: #333;"><strong>📱 연락처:</strong> {row.get('연락처', '-')}</p>
    <p style="margin: 8px 0; color: #333;"><strong>📍 위치:</strong> {row.get('위치', row.get('사무실위치', '-'))}</p>
"""
    
    # 홈페이지를 클릭 가능한 링크로
    homepage = row.get('홈페이지', '-')
    if homepage and homepage != '-' and str(homepage).startswith('http'):
        response += f'    <p style="margin: 8px 0; color: #333;"><strong>🌐 홈페이지:</strong> <a href="{homepage}" target="_blank" style="color: #e83e8c; text-decoration: none;">{homepage} 🔗</a></p>\n'
    else:
        response += f'    <p style="margin: 8px 0; color: #333;"><strong>🌐 홈페이지:</strong> {homepage}</p>\n'
    
    response += "</div>"
    
    return response, "CONTACT_SEARCH"


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
        f"📊 이수 학점: 전필 {completed_required}학점, 전선 {completed_elective}학점 (총 {total_credits}학점)"
    ], "#667eea", "📋")
    
    # 학점 기준 추천
    if total_credits < 20:
        recommendation = "소단위전공과정(마이크로디그리) 또는 부전공"
        reason = "현재 이수 학점이 적어 부담이 적은 제도를 추천드립니다."
    elif total_credits < 40:
        recommendation = "부전공 또는 융합부전공"
        reason = "적절한 학점을 이수하셨습니다. 부전공 도전을 추천드립니다."
    else:
        recommendation = "복수전공 또는 융합전공"
        reason = "충분한 학점을 이수하셨습니다. 복수전공 도전 가능합니다!"
    
    response += f"""
<div style="background: linear-gradient(135deg, #f093fb15 0%, #f5576c15 100%); border-left: 4px solid #f093fb; border-radius: 12px; padding: 16px; margin: 16px 0;">
    <h4 style="margin: 0 0 10px 0; color: #f093fb;">🎯 추천 다전공</h4>
    <p style="font-size: 1.1rem; font-weight: bold; color: #333; margin: 8px 0;">{recommendation}</p>
    <p style="color: #666; font-size: 0.9rem; margin: 8px 0;">💡 {reason}</p>
</div>
"""
    
    response += create_tip_box("왼쪽 '다전공 제도 안내' 메뉴에서 상세 정보를 확인하세요!")
    response += create_contact_box()
    
    return response, "RECOMMENDATION"


def handle_greeting(user_input, extracted_info, data_dict):
    response = create_header_card("안녕하세요!", "👋", "#667eea")
    response += create_simple_card("<p style='margin:0; font-size: 1rem;'><strong>한경국립대학교 다전공 안내 AI챗봇</strong>입니다 😊</p>", "#f0f4ff", "#667eea")
    
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
    response = create_header_card("범위 외 질문", "🚫", "#636e72")
    response += create_simple_card("<p style='margin:0;'>저는 <strong>한경국립대학교 다전공 안내 AI챗봇</strong>이에요.</p>", "#f8f9fa", "#6c757d")
    
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


# 핸들러 매핑 (FAQ로 처리되지 않는 경우 사용)
FALLBACK_HANDLERS = {
    'COURSE_SEARCH': handle_course_search,
    'CONTACT_SEARCH': handle_contact_search,
    'RECOMMENDATION': handle_recommendation,
    'GREETING': handle_greeting,
    'BLOCKED': handle_blocked,
    'OUT_OF_SCOPE': handle_out_of_scope,
    'GENERAL': handle_general,
}


# ============================================================
# 🤖 통합 응답 생성 함수
# ============================================================

def generate_ai_response(user_input, chat_history, data_dict):
    """
    통합 응답 생성 함수
    1. FAQ 매핑 검색 (우선)
    2. Semantic Router + 핸들러
    3. AI Fallback
    """
    faq_df = data_dict.get('faq_mapping', FAQ_MAPPING)
    
    # 1. 의도 분류
    intent, method, extracted_info = classify_intent(user_input)
    
    # 차단된 경우 바로 처리
    if intent == 'BLOCKED':
        return handle_blocked(user_input, extracted_info, data_dict)
    
    # 인사말 처리
    if intent == 'GREETING':
        return handle_greeting(user_input, extracted_info, data_dict)
    
    # 2. FAQ 매핑 검색
    faq_match, score = search_faq_mapping(user_input, faq_df)
    
    if faq_match is not None and score >= 10:
        # FAQ 매칭 성공
        raw_answer = faq_match.get('answer', '')
        program = faq_match.get('program', '')
        
        # AI로 대화체 변환
        conversational_answer = generate_conversational_response(raw_answer, user_input, program)
        
        # HTML 포맷팅
        formatted_response = format_faq_response_html(conversational_answer, program)
        formatted_response += create_contact_box()
        
        return formatted_response, f"FAQ_{faq_match.get('intent', 'UNKNOWN')}"
    
    # 3. 특수 핸들러 필요한 경우 (연락처, 과목 검색, 추천)
    if intent in FALLBACK_HANDLERS:
        return FALLBACK_HANDLERS[intent](user_input, extracted_info, data_dict)
    
    # 4. AI Fallback - 일반 다전공 질문
    try:
        # 프로그램 정보 컨텍스트 생성
        context_parts = []
        programs = data_dict.get('programs', {})
        if programs:
            for prog_name, prog_info in programs.items():
                context_parts.append(f"[{prog_name}]\n- 설명: {prog_info.get('description', '')}\n- 이수학점: {prog_info.get('credits_multi', '')}\n- 신청자격: {prog_info.get('qualification', '')}")
        
        context = "\n\n".join(context_parts[:5])  # 상위 5개만
        
        prompt = f"""당신은 한경국립대학교 다전공 안내 AI챗봇입니다.
다음 정보를 자연스러운 대화체로 변환하세요.

[참고 정보]
{context}

[학생 질문]
{user_input}

[지침]
1. "~합니다", "~해주세요" 등 정중한 종결어미 사용
2. 친근하고 공손한 말투 사용
3. 핵심 정보를 명확하게 전달
4. 모르는 내용은 학사지원팀(031-670-5035) 문의 안내
5. 이모지 적절히 사용 (📅, 📋, ✅ 등)
6. 학사공지 링크: {ACADEMIC_NOTICE_URL}
"""
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config={'temperature': 0.7, 'max_output_tokens': 1000}
        )
        
        ai_response = response.text.strip()
        formatted_response = f"""
<div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea; border-radius: 12px; padding: 16px; margin: 12px 0;">
    {ai_response}
</div>
"""
        formatted_response += create_contact_box()
        
        return formatted_response, "AI_RESPONSE"
        
    except Exception as e:
        return handle_out_of_scope(user_input, extracted_info, data_dict)


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

            edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
            if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                st.caption(f"🏫 운영전공: {str(edu_dept).strip()}")


def display_courses(major, program_type):
    """과목 정보 표시"""
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
                    st.markdown(f"#### 📅 {semester}학기")
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
                    else:
                        for _, row in semester_courses.iterrows():
                            course_name = row.get('과목명', '')
                            credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                            edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                            if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                                st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                            else:
                                st.write(f"• {course_name} ({credit})")
                    
                    st.divider()
            else:
                st.markdown("**📚 교과목 목록**")
                for _, row in courses.iterrows():
                    course_name = row.get('과목명', '')
                    credit = f"{int(row.get('학점', 0))}학점" if pd.notna(row.get('학점')) else ""
                    edu_dept = row.get('교과목 운영전공') or row.get('교과목운영전공', '')
                    if is_micro and pd.notna(edu_dept) and str(edu_dept).strip():
                        st.write(f"• {course_name} ({credit}, {str(edu_dept).strip()})")
                    else:
                        st.write(f"• {course_name} ({credit})")
        
        st.markdown("---")
        display_major_contact(display_major, program_type)
        return True
    else:
        st.info(f"'{display_major}' 교과목 정보가 없습니다.")
        return False


def display_major_contact(major, program_type="전공"):
    """전공 연락처 표시"""
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
    if edu_major and ("소단위" in program_type or "마이크로" in program_type):
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == edu_major]
        if contact_row.empty:
            contact_row = MAJORS_INFO[MAJORS_INFO['교육운영전공'] == edu_major]
    
    if contact_row.empty:
        contact_row = MAJORS_INFO[MAJORS_INFO['전공명'] == clean_major]
    
    if contact_row.empty:
        keyword = clean_major.replace('전공', '').replace('과정', '').replace('(', '').replace(')', '')[:4]
        if keyword:
            contact_row = MAJORS_INFO[MAJORS_INFO['전공명'].str.contains(keyword, na=False, regex=False)]
    
    if not contact_row.empty:
        row = contact_row.iloc[0]
        
        if "소단위" in program_type or "마이크로" in program_type:
            if edu_major:
                major_name = edu_major
            elif pd.notna(row.get('교육운영전공')) and str(row.get('교육운영전공')).strip():
                major_name = str(row.get('교육운영전공')).strip()
            else:
                major_name = row.get('전공명', major)
        else:
            major_name = row.get('전공명', major)
        
        phone = row.get('연락처', '')
        location = row.get('사무실위치', row.get('위치', ''))
        
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


def render_question_buttons(questions, key_prefix, cols=5):
    btn_cols = st.columns(cols)
    for i, q in enumerate(questions):
        if btn_cols[i % cols].button(q, key=f"{key_prefix}_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response_text, res_type = generate_ai_response(q, st.session_state.chat_history[:-1], ALL_DATA)
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
            st.rerun()


# ============================================================
# 🖥️ 메인 UI
# ============================================================

def main():
    initialize_session_state()
    
    # 사이드바
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0;'>🎓</h1>
            <h3 style='margin-top: 0;'>HKNU 다전공</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # FAQ 메뉴 삭제 - 2개 메뉴만 유지
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내"], 
            icons=["chat-dots-fill", "journal-bookmark-fill"],
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
                한경국립대 다전공 제도에 관한<br>
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
        st.subheader("💬 AI챗봇과 대화하기")
        
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
                "다전공 제도 비교해줘",
                "복수전공은 뭐야?",
                "마이크로디그리 알려줘?",
                "복수·부전공 차이는?",
            ]
            render_question_buttons(q_program, "qp", cols=2)

        with tab_credit:
            q_credit = [
                "다전공별 이수학점은?",
                "복수전공 학점은?",
            ]
            render_question_buttons(q_credit, "qc", cols=2)

        with tab_etc:
            q_etc = [
                "경영학전공 연락처 알려줘",
                "응용수학전공 사무실은 어디야?",
                "기계공학전공 교과목은?",
            ]
            render_question_buttons(q_etc, "qe", cols=2)

        st.divider()
        
        # 채팅 히스토리 표시
        for chat in st.session_state.chat_history:
            avatar = "🧑‍🎓" if chat["role"] == "user" else "🤖"
            with st.chat_message(chat["role"], avatar=avatar):
                st.markdown(chat["content"], unsafe_allow_html=True)
        
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
        <h1 style="font-size: 2rem; margin-bottom: 20px; color: #1f2937;">
            📊 제도 한눈에 비교
        </h1>
        """, unsafe_allow_html=True)
        
        # 제도 비교 카드
        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '')[:50] + '...' if len(info.get('description', '')) > 50 else info.get('description', '-')
                    qual = info.get('qualification', '-')[:30] + '...' if len(str(info.get('qualification', '-'))) > 30 else info.get('qualification', '-')
                    
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
                    return "융합전공" in type_str and "융합부전공" not in type_str
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
                is_special = any(sp in selected_program for sp in ["융합전공", "융합부전공", "소단위", "마이크로"])
                
                if is_special:
                    category_majors = {"전체": sorted(available_majors.keys())}
                else:
                    category_majors = get_majors_by_category(selected_program)
                
                if selected_program in target_programs:
                    if is_special or len(category_majors) <= 1:
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
                    field_majors = {}
                    major_to_edu_major = {}
                    
                    if not MAJORS_INFO.empty:
                        mask = MAJORS_INFO['제도유형'].apply(lambda x: any(kw in str(x).lower() for kw in ['소단위', '마이크로', 'md']))
                        micro_df = MAJORS_INFO[mask]
                        
                        group_column = None
                        if '분야' in MAJORS_INFO.columns:
                            group_column = '분야'
                        elif '계열' in MAJORS_INFO.columns:
                            group_column = '계열'
                        
                        for _, row in micro_df.iterrows():
                            if group_column:
                                field = row.get(group_column, '기타')
                                if pd.isna(field) or str(field).strip() == '':
                                    field = '기타'
                                field = str(field).strip()
                            else:
                                field = '전체'
                            
                            major_name = row['전공명']
                            edu_major = row.get('교육운영전공', '')
                            
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
                st.warning(f"⚠️ {selected_program}에 해당하는 전공 목록을 찾을 수 없습니다.")
                st.info("💡 데이터 파일에 해당 제도의 전공 정보가 있는지 확인해주세요.")


if __name__ == "__main__":
    initialize_session_state()
    main()
