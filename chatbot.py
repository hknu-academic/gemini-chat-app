import streamlit as st
from google import genai
import pandas as pd
from streamlit_option_menu import option_menu 
from datetime import datetime
import os
from dotenv import load_dotenv  # 설치 필요
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import re

# === [AI 설정] Gemini API 연결 ===
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다!")
    st.info("Streamlit Secrets에 API 키를 저장해주세요.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)  # <--- Client 객체 생성 방식으로 변경

# === 페이지 설정 (가장 먼저 실행되어야 함) ===
st.set_page_config(
    page_title="다전공 안내 AI챗봇",
    page_icon="🎓",
    layout="wide",
)

# === 자동 스크롤 함수 (마지막 말풍선 추적 방식 + Focus) ===
def scroll_to_bottom():
    # 매번 새로운 ID로 강제 실행 유도
    unique_id = str(uuid.uuid4())
    
    js = f"""
    <script>
        // Random ID to force update: {unique_id}
        
        function scrollIntoView() {{
            // 1. 말풍선 요소들을 다 찾습니다.
            var messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            
            if (messages.length > 0) {{
                // 2. 가장 마지막 말풍선을 가져옵니다.
                var lastMessage = messages[messages.length - 1];
                
                // 3. 그 말풍선이 보이도록 화면을 부드럽게 내립니다.
                lastMessage.scrollIntoView({{behavior: "smooth", block: "end"}});
            }} else {{
                // 말풍선을 못 찾으면 기존 방식으로 컨테이너 스크롤 시도
                var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) container.scrollTop = container.scrollHeight;
            }}
        }}

        // 화면 렌더링 시간을 고려해 조금 넉넉히 기다렸다가 실행
        setTimeout(scrollIntoView, 300);
        setTimeout(scrollIntoView, 500);
    </script>
    """
    st.components.v1.html(js, height=0)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'scroll_to_bottom' not in st.session_state:
    st.session_state.scroll_to_bottom = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = {}
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if "scroll_count" not in st.session_state:
    st.session_state.scroll_count = 0
if 'show_calculator' not in st.session_state:
    st.session_state.show_calculator = False

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'chat_history': [],
        'user_info': {},
        'feedback_data': [],
        'is_admin': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === 데이터 로드 함수 ===
@st.cache_data
def load_all_data():
    """모든 엑셀 데이터를 로드하여 딕셔너리로 반환"""
    data = {}
    try:
        data['programs'] = load_programs()
        data['faq'] = load_faq()
        data['curriculum'] = load_curriculum_mapping()
        data['courses'] = load_courses()
        data['keywords'] = load_keywords()
        data['grad_req'] = load_graduation_requirements()
        data['primary_req'] = load_primary_requirements()
        data['majors'] = load_majors_info()
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
    return data

@st.cache_data
def load_programs():
    """제도 정보 로드"""
    try:
        df = pd.read_excel('data/programs.xlsx')
        programs = {}
        for _, row in df.iterrows():
            programs[row['제도명']] = {
                'description': row['설명'],
                'credits_general': row['이수학점(교양)'] if pd.notna(row.get('이수학점(교양)')) else '-',
                'credits_primary': row['원전공 이수학점'] if pd.notna(row.get('원전공 이수학점')) else '-',
                'credits_multi': row['다전공 이수학점'] if pd.notna(row.get('다전공 이수학점')) else '-',
                'graduation_certification': row['졸업인증'] if pd.notna(row.get('졸업인증')) else '-',
                'graduation_exam': row['졸업시험'] if pd.notna(row.get('졸업시험')) else '-',
                'qualification': row['신청자격'],
                'degree': row['학위기 표기'],
                'difficulty': '★' * int(row['난이도']) + '☆' * (5 - int(row['난이도'])),
                'features': row['특징'].split(',') if pd.notna(row.get('특징')) else [],
                'notes': row['기타'] if pd.notna(row.get('기타')) else ''
            }
        return programs
    except FileNotFoundError:
        st.warning("⚠️ data/programs.xlsx 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        return get_sample_programs()
    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {e}")
        return get_sample_programs()

@st.cache_data
def load_faq():
    """FAQ 로드"""
    try:
        df = pd.read_excel('data/faq.xlsx')
        return df.to_dict('records')
    except FileNotFoundError:
        st.warning("⚠️ data/faq.xlsx 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        return get_sample_faq()
    except Exception as e:
        st.error(f"❌ FAQ 로드 오류: {e}")
        return get_sample_faq()

@st.cache_data
def load_curriculum_mapping():
    """이수체계도 이미지 매핑 로드"""
    try:
        df = pd.read_excel('data/curriculum_mapping.xlsx')
        return df
    except FileNotFoundError:
        st.warning("⚠️ data/curriculum_mapping.xlsx 파일을 찾을 수 없습니다.")
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])
    except Exception as e:
        st.error(f"❌ 매핑 데이터 로드 오류: {e}")
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])

@st.cache_data
def load_courses():
    """과목 정보 로드"""
    try:
        df = pd.read_excel('data/courses.xlsx')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])
    except Exception as e:
        st.error(f"❌ 과목 데이터 로드 오류: {e}")
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])

@st.cache_data
def load_keywords():
    """키워드 매핑 로드"""
    try:
        df = pd.read_excel('data/keywords.xlsx')
        return df.to_dict('records')
    except FileNotFoundError:
        st.warning("⚠️ data/keywords.xlsx 파일을 찾을 수 없습니다. 기본 키워드를 사용합니다.")
        return get_default_keywords()
    except Exception as e:
        st.error(f"❌ 키워드 로드 오류: {e}")
        return get_default_keywords()

@st.cache_data
def load_graduation_requirements():
    """졸업요건(기준학번별 학점) 로드"""
    try:
        df = pd.read_excel('data/graduation_requirements.xlsx')
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 졸업요건 로드 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def load_primary_requirements():
    """본전공 이수요건 데이터 로드"""
    try:
        df = pd.read_excel('data/primary_requirements.xlsx')
        if not df.empty:
            cols = ['전공명', '구분']
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_majors_info():
    """전공 정보 로드 (연락처, 홈페이지 포함)"""
    try:
        df = pd.read_excel('data/majors_info.xlsx')
        return df
    except FileNotFoundError:
        st.warning("⚠️ data/majors_info.xlsx 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 전공 정보 로드 오류: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)  # 캐시 유지, 필요 시 제거 가능
def create_faq_vectorizer(force_reload=False):
    """FAQ 질문들을 벡터화"""
    if force_reload:
        create_faq_vectorizer.clear()
    
    questions = [faq['질문'] for faq in FAQ_DATA if faq.get('질문')]
    vectorizer = TfidfVectorizer()
    
    if questions:
        vectors = vectorizer.fit_transform(questions)
        return vectorizer, vectors, questions
    return None, None, []


def get_default_keywords():
    """기본 키워드 데이터"""
    return [
        {"키워드": "복수전공", "타입": "제도", "연결정보": "복수전공"},
        {"키워드": "복전", "타입": "제도", "연결정보": "복수전공"},
        {"키워드": "부전공", "타입": "제도", "연결정보": "부전공"},
        {"키워드": "부전", "타입": "제도", "연결정보": "부전공"},
        {"키워드": "연계전공", "타입": "제도", "연결정보": "연계전공"},
        {"키워드": "융합전공", "타입": "제도", "연결정보": "융합전공"},
        {"키워드": "융합부전공", "타입": "제도", "연결정보": "융합부전공"},
        {"키워드": "마이크로디그리", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "마디", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "MD", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "학점", "타입": "주제", "연결정보": "학점정보"},
        {"키워드": "이수학점", "타입": "주제", "연결정보": "학점정보"},
        {"키워드": "신청", "타입": "주제", "연결정보": "신청정보"},
        {"키워드": "지원", "타입": "주제", "연결정보": "신청정보"},
        {"키워드": "비교", "타입": "주제", "연결정보": "비교표"},
        {"키워드": "차이", "타입": "주제", "연결정보": "비교표"},
        {"키워드": "졸업", "타입": "주제", "연결정보": "졸업요건"},
        {"키워드": "졸업인증", "타입": "주제", "연결정보": "졸업요건"},
        {"키워드": "졸업시험", "타입": "주제", "연결정보": "졸업요건"},
    ]

def get_sample_programs():
    """샘플 제도 데이터"""
    return {
        "복수전공": {
            "description": "주전공 외에 다른 전공을 추가로 이수하여 2개의 학위를 취득하는 제도",
            "credits_general": "-",
            "credits_major": "36학점 이상",
            "graduation_certification": "불필요",
            "graduation_exam": "불필요",
            "qualification": "2학년 이상, 평점 2.0 이상",
            "degree": "2개 학위 수여",
            "difficulty": "★★★★☆",
            "features": ["졸업 시 2개 학위 취득", "취업 시 경쟁력 강화", "학점 부담 높음"],
            "notes": ""
        },
        "부전공": {
            "description": "주전공 외에 다른 전공의 기초과목을 이수하는 제도",
            "credits_general": "-",
            "credits_major": "21학점 이상",
            "graduation_certification": "불필요",
            "graduation_exam": "불필요",
            "qualification": "2학년 이상",
            "degree": "주전공 학위 (부전공 표기)",
            "difficulty": "★★☆☆☆",
            "features": ["학점 부담 적음", "학위증에 부전공 표기"],
            "notes": ""
        }
    }

def get_sample_faq():
    """샘플 FAQ 데이터"""
    return [
        {
            "카테고리": "일반",
            "질문": "복수전공과 부전공의 차이는?",
            "답변": "복수전공은 36학점 이상을 이수하여 2개의 학위를 받지만, 부전공은 21학점 이수로 주전공 학위만 받습니다."
        }
    ]

# 데이터 로드
PROGRAM_INFO = load_programs()
FAQ_DATA = load_faq()
CURRICULUM_MAPPING = load_curriculum_mapping()
COURSES_DATA = load_courses()
KEYWORDS_DATA = load_keywords()
GRAD_REQUIREMENTS = load_graduation_requirements()
PRIMARY_REQUIREMENTS = load_primary_requirements()
MAJORS_INFO = load_majors_info()  # 🆕 전공 정보 로드

def token_partial_match(root_input, target_clean):
    """
    root_input: 정제된 사용자 입력 (공백 제거된 상태)
    target_clean: 전공명 정제 문자열
    """
    # 한글/영문 토큰 추출
    tokens = re.findall(r'[가-힣a-zA-Z]+', root_input)

    for t in tokens:
        if len(t) >= 2 and t in target_clean:
            return True
    return False

def normalize_major_type(val):
    v = str(val)
    if '필수' in v or '전필' in v:
        return '전공필수'
    if '선택' in v or '전선' in v:
        return '전공선택'
    return '기타'

# 🆕 NEW: 동의어 처리 함수 추가
def normalize_query(text):
    """질문 정규화 및 동의어 처리"""
    text_lower = text.lower().replace(" ", "")
    
    synonyms = {
        "언제": "기간",
        "몇시": "기간",
        "얼마": "학점",
        "어떻게": "방법",
    }
    
    for original, replacement in synonyms.items():
        text_lower = text_lower.replace(original, replacement)
    
    text_clean = re.sub(r'(이야|야|에요|예요|까|니|냐|나요)', '', text_lower)
    return text_clean

# === [핵심] AI 지식 검색 함수 (RAG) - 수정 버전 ===
def get_ai_context(user_input, data_dict):
    context = ""
    user_input_clean = user_input.replace(" ", "").lower()

    # ✅ 반드시 먼저 초기화
    is_course_query = False
    is_contact_query = False
    is_md_query = False
    
    is_course_query = any(w in user_input_clean for w in ["과목", "교과목", "추천", "리스트", "수강", "학년", "배워"])
    is_contact_query = any(w in user_input_clean for w in ["연락처", "사무실", "위치", "번호", "전화", "홈페이지", "사이트"])
    is_md_query = any(w in user_input_clean for w in ['md', '마이크로', '소단위', '마디'])

    # 데이터 가져오기 (전역 변수 활용 및 안전장치)
    majors_info = data_dict.get('majors', MAJORS_INFO)
    primary_req = data_dict.get('primary_req', PRIMARY_REQUIREMENTS)
    courses_data = data_dict.get('courses', COURSES_DATA)
    faq_data = data_dict.get('faq', FAQ_DATA)
    prog_info = data_dict.get('programs', PROGRAM_INFO)


    # 🔥 핵심 수정: 전공명 추출 로직 개선

        
    # 1️⃣ 불필요한 단어 제거
    raw_keyword = re.sub(r'[^\w]', '', user_input_clean)

    if is_course_query:
        # "법학전공과목알려줘" -> "법학"
        root_input = re.sub(
            r'(알려줘|보여줘|뭐있어|어디야|어디에|어디|뭐야|전공|학과|학부|과목|교과목|리스트|추천|해줘|학년|은|는|이|가|을|를|의)',
            '',
            raw_keyword
        )

    elif is_contact_query:
        # "법학전공연락처" -> "법학"
        root_input = re.sub(
            r'(알려줘|보여줘|어디에|어디야|어디|뭐야|뭐있어|전공|학과|학부|연락처|사무실|위치|번호|전화|홈페이지|은|는|이|가|을|를|의)',
            '',
            raw_keyword
        )
    elif is_md_query:
        root_input = re.sub(
            r'(알려줘|보여줘|뭐있어|어디야|어디에|어디|뭐야|교과목|리스트|과목|추천|해줘|은|는|이|가|을|를|의)',
            '',
            raw_keyword
        )
    else:
        root_input = re.sub(
            r'(알려줘|보여줘|어디야|어디에|어디|뭐야|전공|학과|학부|과목|신청|학점|은|는|이|가|을|를|의)',
            '',
            raw_keyword
        )
        

    # 전공 목록 확보
    all_majors_set = set()
    if not majors_info.empty and '전공명' in majors_info.columns:
        all_majors_set.update(majors_info['전공명'].dropna().astype(str).unique())
    if not courses_data.empty and '전공명' in courses_data.columns:
        all_majors_set.update(courses_data['전공명'].dropna().astype(str).unique())
    
    major_list = list(all_majors_set)
    
    # 3️⃣ 학년 추출
    target_year = None
    for i in range(1, 5):
        if f"{i}학년" in user_input_clean:
            target_year = i
            break

    # 4️⃣ 🔥 전공 매칭 (핵심 수정!)
    matched_majors = set()
    
    # 🔧 디버깅: 입력값 확인
    debug_info = f"[디버그] root_input='{root_input}', 전공수={len(major_list)}\n"
    
    for m_str in major_list:
        m_clean = re.sub(r'\s+', '', m_str).lower()
        m_root = m_clean.replace("전공", "").replace("학과", "").replace("학부", "")
        
        # 🔧 디버깅: 시각미디어디자인 관련만 로그
        if "시각" in m_clean or "미디어" in m_clean or "디자인" in m_clean:
            debug_info += f"  - 검사중: {m_str} (m_clean={m_clean}, m_root={m_root})\n"
        
        # 매칭 조건 (우선순위 순서)
        # 1순위: 정확히 포함
        if root_input in m_clean or root_input in m_root:
            matched_majors.add(m_str)
            if "시각" in m_clean or "미디어" in m_clean or "디자인" in m_clean:
                debug_info += f"    ✅ 매칭 성공 (1순위)\n"
            continue
        
        # 2순위: 핵심 단어만 일치 (예: "법학" in "법학전공")
        if len(root_input) >= 2:
            # "법학" -> "법학전공", "법학과" 모두 매칭
            if m_root.startswith(root_input) or root_input in m_root:
                matched_majors.add(m_str)
                if "시각" in m_clean or "미디어" in m_clean or "디자인" in m_clean:
                    debug_info += f"    ✅ 매칭 성공 (2순위)\n"
                continue
        
        # 3순위: 부분 토큰 매칭
        if len(root_input) >= 3 and token_partial_match(root_input, m_clean):
            matched_majors.add(m_str)
            if "시각" in m_clean or "미디어" in m_clean or "디자인" in m_clean:
                debug_info += f"    ✅ 매칭 성공 (3순위)\n"
    
    # 🔧 디버깅 정보를 context에 임시 추가
    if "시각" in user_input_clean and not matched_majors:
        context += f"\n### [🔧 디버깅 정보]\n{debug_info}\n"

    # 5️⃣ 과목 정보 추가 (matched_majors가 있을 때만)

    if matched_majors:
        context += f"[검색된 특정 전공: {', '.join(matched_majors)}]\n\n"

        for m_name in list(matched_majors)[:10]:
            # A. 기본 정보
            if not majors_info.empty:
                m_rows = majors_info[majors_info['전공명'] == m_name]
                if not m_rows.empty:
                    m_row = m_rows.iloc[0]
                    p_type = str(m_row.get('제도유형', ''))

                    context += f"### [{m_name} 상세정보]\n"
                    context += f"- 제도유형: {p_type}\n"
                    context += f"- 소개: {m_row.get('전공설명','-')}\n"

                    # 🆕 마이크로디그리인 경우 교육운영전공(주관부서) 정보 찾기
                    if '마이크로' in p_type or '소단위' in p_type or 'MD' in m_name:
                        op_major = m_row.get('교육운영전공', '-')
                        if op_major != '-' and pd.notna(op_major):
                            context += f"- ★교육운영전공(주관부서): {op_major}\n"
                            op_row = majors_info[majors_info['전공명'] == op_major]
                            if not op_row.empty:
                                op_row_data = op_row.iloc[0]
                                context += f"  - 연락처: {op_row_data.get('연락처','-')}\n"
                                context += f"  - 위치: {op_row_data.get('위치','-')}\n"
                                context += f"  - 홈페이지: {op_row_data.get('홈페이지','-')}\n"
                    else:
                        context += f"- 연락처: {m_row.get('연락처','-')}\n"
                        context += f"- 위치: {m_row.get('위치','-')}\n"
                        context += f"- 홈페이지: {m_row.get('홈페이지','-')}\n"

                    context += "\n"
            
            # B. 과목 정보 (is_course_query일 때만)
            if not courses_data.empty and is_course_query:
                # 🔥 수정: 부분 일치로 검색
                major_courses = courses_data[
                    courses_data['전공명'].str.contains(m_name, case=False, na=False, regex=False)
                ]

                # MD가 아니면 마이크로디그리 과목 제외
                if not is_md_query and '제도유형' in major_courses.columns:
                    major_courses = major_courses[
                        ~major_courses['제도유형'].astype(str)
                        .str.contains('소단위|마이크로|MD', case=False, na=False)
                    ]

                # 학년 필터
                if target_year:
                    major_courses = major_courses[
                        (major_courses['학년'] == target_year) |
                        (major_courses['학년'].astype(str) == str(target_year)) |
                        major_courses['학년'].isna()
                    ]

                if major_courses.empty:
                    context += f"[안내] {m_name} {target_year if target_year else ''}학년 과목 정보가 데이터에 없습니다.\n"
                else:
                    # ✅ 이수구분 정규화
                    major_courses['이수구분_정리'] = major_courses['이수구분'].apply(normalize_major_type)
                    
                    # ✅ 학년별로 그룹화해서 출력
                    context += f"### [{m_name} 교과목 안내]\n\n"
                    
                    # 학년 정렬 (1, 2, 3, 4, NaN 순서)
                    years = []
                    for y in [1, 2, 3, 4]:
                        if y in major_courses['학년'].values or str(y) in major_courses['학년'].astype(str).values:
                            years.append(y)
                    
                    for year in years:
                        year_courses = major_courses[
                            (major_courses['학년'] == year) | 
                            (major_courses['학년'].astype(str) == str(year))
                        ]
                        
                        if year_courses.empty:
                            continue
                        
                        context += f"#### {year}학년\n"
                        
                        # 전공필수 먼저
                        required = year_courses[year_courses['이수구분_정리'] == '전공필수']
                        if not required.empty:
                            context += "🔴 **전공필수**\n"
                            for _, row in required.iterrows():
                                semester = row.get('학기', '-')
                                course_name = row.get('과목명', '-')
                                credit = row.get('학점', '-')
                                context += f"  - {semester}학기: {course_name} ({credit}학점)\n"
                            context += "\n"
                        
                        # 전공선택
                        elective = year_courses[year_courses['이수구분_정리'] == '전공선택']
                        if not elective.empty:
                            context += "🟢 **전공선택**\n"
                            for _, row in elective.iterrows():
                                semester = row.get('학기', '-')
                                course_name = row.get('과목명', '-')
                                credit = row.get('학점', '-')
                                context += f"  - {semester}학기: {course_name} ({credit}학점)\n"
                            context += "\n"
                        
                        # 기타
                        others = year_courses[year_courses['이수구분_정리'] == '기타']
                        if not others.empty:
                            context += "🔵 **기타**\n"
                            for _, row in others.iterrows():
                                semester = row.get('학기', '-')
                                course_name = row.get('과목명', '-')
                                credit = row.get('학점', '-')
                                isutype = row.get('이수구분', '-')
                                context += f"  - {semester}학기: {course_name} ({credit}학점) [{isutype}]\n"
                            context += "\n"
                    
                    context += "\n"


    # ==========================================================
    # [2] 마이크로디그리 전용 추가 검색 (is_md_query일 때만)
    # ==========================================================
    if is_md_query and not courses_data.empty:
        if '제도유형' in courses_data.columns and '전공명' in courses_data.columns:
            
            # MD 과목만 필터링
            md_courses_df = courses_data[
                courses_data['제도유형'].astype(str).str.contains('소단위|마이크로|MD', case=False, na=False)
            ]
            
            if not md_courses_df.empty:
                md_major_list = md_courses_df['전공명'].unique()
                matched_md_majors = []

                for m_name in md_major_list:
                    m_str = str(m_name)
                    m_clean = re.sub(r'^\d+\.?\s*', '', m_str)
                    m_clean = re.sub(r'[^\w]', '', m_clean.lower())
                    
                    paren_match = re.search(r'\(([^)]+)\)', m_str)
                    parent_major = ""
                    if paren_match:
                        parent_major = re.sub(r'[^\w]', '', paren_match.group(1).lower())
                    
                    match_found = False
                    
                    if root_input in m_clean or m_clean in root_input:
                        match_found = True
                    elif token_partial_match(root_input, m_clean):
                        match_found = True
                    elif parent_major and token_partial_match(root_input, parent_major):
                        match_found = True
                    elif len(root_input) >= 4 and root_input[:4] in m_clean:
                        match_found = True
                    
                    if match_found and m_name not in matched_majors:
                        matched_md_majors.append(m_name)

                # MD 매칭 결과 추가
                if matched_md_majors:
                    for m_name in matched_md_majors:
                        context += f"### [🎯 {m_name} 과목 리스트]\n"
                        context += "※ 이 과목들은 '소단위전공과정(마이크로디그리)' 이수용 과목입니다.\n"
                        
                        m_courses = md_courses_df[md_courses_df['전공명'] == m_name]
                        for _, row in m_courses.head(25).iterrows():
                            term = row.get('학기', '-')
                            try:
                                term = int(float(term))
                                term_str = f"{term}학기"
                            except:
                                term_str = ""
                            
                            # 🔥 마이크로디그리는 학년 표시 제외
                            if term_str:
                                context += f"- {term_str}: {row['과목명']} ({row['학점']}학점)\n"
                            else:
                                context += f"- {row['과목명']} ({row['학점']}학점)\n"
                        context += "\n"

    # ==========================================================
    # [3] 🆕 신청 자격 요건 비교 (모든 제도)
    # ==========================================================
    qualification_keywords = ["신청자격", "지원자격", "요건", "자격요건", "신청요건"]
    if any(kw in user_input_clean for kw in qualification_keywords):
        context += "\n### [📋 다전공 제도별 신청 자격 요건 비교]\n\n"
        context += "| 제도 | 신청 자격 |\n"
        context += "|------|----------|\n"
        
        for p_name, p_info in prog_info.items():
            qual = p_info.get('qualification', '-')
            context += f"| {p_name} | {qual} |\n"
        
        context += "\n💡 **참고**: 신청 자격은 학칙 개정에 따라 변경될 수 있으니, 신청 전 학사지원팀(031-670-5035)에 확인하세요.\n\n"

    # ==========================================================
    # [4] 제도 카테고리 리스트 (특정 전공이 없을 때만)
    # ==========================================================
    if not matched_majors:
        categories = {
            "융합전공": ["융합전공", "융합"],
            "부전공": ["부전공"],
            "복수전공": ["복수전공", "복전"],
            "마이크로디그리": ["마이크로디그리", "마디", "소단위", "md"],
            "연계전공": ["연계전공", "연계"]
        }

        for cat_name, keywords in categories.items():
            if any(kw in user_input_clean for kw in keywords):
                if not majors_info.empty and '제도유형' in majors_info.columns:
                    matched_rows = majors_info[majors_info['제도유형'].str.contains(cat_name, na=False)]
                    if not matched_rows.empty:
                        major_list = matched_rows['전공명'].tolist()
                        context += f"[{cat_name} 전체 목록]\n- {', '.join(major_list)}\n\n"

    # ==========================================================
    # [5] 본전공 이수요건 검색
    # ==========================================================
    if not primary_req.empty:
        pm_input = re.sub(r'(전공|학과|학부|의|신청|학점|알려줘|md)', '', user_input_clean)
        matched_primary = [m for m in primary_req['전공명'].unique() if pm_input in str(m).lower()]
        
        for m in matched_primary[:1]:
            df_major = primary_req[primary_req['전공명'] == m]
            context += f"### [{m}] 본전공 이수학점 상세 기준\n"
            for _, row in df_major.iterrows():
                context += f"- 구분: {row['구분']}, 본전공필수: {row.get('본전공_전필',0)}, 전공선택: {row.get('본전공_전선',0)}, 계: {row.get('본전공_계',0)}\n"

    # ==========================================================
    # [6] FAQ 검색
    # ==========================================================
    if faq_data:
        for faq in faq_data:
            if user_input_clean in str(faq['질문']).replace(" ","").lower():
                context += f"[FAQ] Q: {faq['질문']}\nA: {faq['답변']}\n\n"

    # ==========================================================
    # [7] 제도 자체 설명
    # ==========================================================
    for p_name, p_info in prog_info.items():
        if p_name in user_input_clean:
            context += f"### [{p_name}] 제도 설명\n- {p_info['description']}\n- 이수학점: {p_info['credits_multi']}\n\n"

    return context

def analyze_query_type(user_input):
    """질문 유형 자동 분류"""
    u = user_input.lower()
    
    if any(w in u for w in ["과목", "추천", "수강", "교과목", "강의"]):
        return "course_recommendation"
    elif any(w in u for w in ["학점", "이수"]):
        return "credit_inquiry"
    elif any(w in u for w in ["연락처", "사무실", "번호", "전화", "위치"]):
        return "contact"
    else:
        return "general"
    

def build_smart_prompt(user_input, context, faq_context, history):
    """질문 유형에 따라 최적화된 프롬프트 생성"""
    
    query_type = analyze_query_type(user_input)
    
    base_prompt = """당신은 한경국립대학교 다전공 안내 전문가입니다.

[핵심 원칙]
1. 제공된 데이터 우선 활용
2. 없는 정보는 솔직히 인정 → 담당 부서 안내
3. 숫자는 **강조**, 친절한 습니다체 사용
4. 답변 끝에 #키워드 2-3개 추가

[데이터]
{context if context else "검색 결과 없음"}

[관련 FAQ]
{faq_context}

[대화 기록]
{history}

질문: {user_input}
"""
    # 질문 유형별 가이드
    if query_type == "course_recommendation":
        specific_guide = """
[과목 추천 시]
- 학년/학기 정보 포함
- 전공필수(🔴)와 전공선택(🟢) 구분
- 추천 이유 간단히 설명
"""
    
    elif query_type == "credit_inquiry":
        specific_guide = """
[학점 안내 시]
- 표 형식으로 정리
- 본전공 변동 학점 반드시 포함
"""
    
    elif query_type == "contact":
        specific_guide = """
[연락처 안내 시]
- 표 형식 사용
- 마이크로디그리는 주관 전공 사무실 안내
"""
    
    else:
        specific_guide = """
[일반 안내]
- 모호한 질문은 후보 제시 후 되묻기
"""
    
    return base_prompt + guide


# === FAQ 스마트 검색 함수 ===
def get_faq_smart_search(user_input, top_n=3):
    """키워드 + 유사도 결합 FAQ 검색 (개선 버전)"""
    user_normalized = normalize_query(user_input)  # 🆕
    user_lower = user_input.lower().replace(" ", "")  # 원본 소문자 버전

    keyword_matches = []

    negative_keywords = {
        "포기_취소": ["포기", "취소", "철회", "그만", "중단"],
        "변경_수정": ["변경", "수정", "바꾸", "전환"],
        "문제_오류": ["오류", "에러", "안돼", "실패"],
        "연기": ["연기", "미루"],
    }

    # 필터링 헬퍼 함수
    def is_filtered_by_negative(faq_question, user_input_text):
        faq_clean = faq_question.replace(" ", "")
        for _, keywords in negative_keywords.items():
            if any(k in faq_clean for k in keywords): # FAQ에 네거티브 단어가 있고
                if not any(k in user_input_text for k in keywords): # 사용자 질문엔 없다면
                    return True # 필터링 대상
        return False
    
    # 🆕 사용자 질문에 특수 키워드(포기, 취소 등)가 있는지 확인
    has_special_keywords = any(k in user_lower for k in ["포기", "취소", "철회", "그만", "중단", "변경", "수정"])
    
    # ===================================================
    # 1️⃣ 핵심 키워드 조합 매칭 (우선순위 높음)
    # ===================================================
    keyword_combinations = [
        # [사용자가 쓸 키워드들, FAQ에 있을 키워드들, 가중치]
        # 🔥 포기/취소 관련을 최우선으로 배치
        (["포기", "취소", "철회", "중단"], ["포기", "취소", "철회"], 0.98),
        (["변경", "수정", "전환"], ["변경", "수정", "전환"], 0.95),
        (["신청기간", "지원기간", "접수기간"], ["신청", "기간", "지원", "접수"], 0.95),
        (["신청방법", "지원방법", "신청절차"], ["신청", "방법", "절차"], 0.90),
        (["신청자격", "지원자격", "자격요건"], ["신청", "자격", "요건"], 0.92),  # 🆕 추가
        (["학점", "몇학점", "이수학점"], ["학점", "이수"], 0.85),
        (["차이", "비교", "다른점"], ["차이", "비교"], 0.85),
    ]

    for user_keywords, faq_keywords, score in keyword_combinations:
        # 사용자 입력에 키워드 조합이 있는지 확인
        if any(uk in user_lower for uk in user_keywords):
            for faq in FAQ_DATA:
                if is_filtered_by_negative(faq['질문'], user_lower): continue

                faq_text = (faq['질문'] + " " + faq['답변']).lower().replace(" ", "")
                
                # FAQ에 관련 키워드들이 있는지 확인
                matches = sum(1 for fk in faq_keywords if fk in faq_text)
                
                # 2개 이상 매칭되면 추가
                if matches >= 2:
                    keyword_matches.append({
                        'faq': faq,
                        'similarity': score,
                        'match_type': 'keyword_combo'
                    })
    
    # 2️⃣ 단일 키워드 매칭 (중간 우선순위)
    # ===================================================
    # 🔥 수정: 특수 키워드가 있으면 일반 신청 키워드 매칭 건너뛰기
    period_keywords = ["신청", "지원", "기간", "언제", "마감", "접수", "시작", "종료"]
    is_period_query = any(kw in user_lower for kw in period_keywords)
    
    if is_period_query and not has_special_keywords:  # 🔥 조건 추가
        for faq in FAQ_DATA:
            faq_text = (faq['질문'] + " " + faq['답변']).lower().replace(" ", "")
            
            # FAQ에 관련 키워드가 있는지
            if any(kw in faq_text for kw in period_keywords):
                # 이미 추가된 FAQ가 아니면 추가
                if not any(m['faq']['질문'] == faq['질문'] for m in keyword_matches):
                    keyword_matches.append({
                        'faq': faq,
                        'similarity': 0.80,
                        'match_type': 'keyword_single'
                    })
    
    # 3️⃣ TF-IDF 유사도 검색 (백업)
    # ===================================================
    similarity_matches = get_top_similar_faqs(user_input, top_n=top_n * 2)
    
    # 4️⃣ 결합 및 중복 제거
    # ===================================================
    all_matches = keyword_matches + similarity_matches
    seen_questions = set()
    unique_matches = []
    
    for match in all_matches:
        q = match['faq']['질문']
        if q not in seen_questions:
            seen_questions.add(q)
            unique_matches.append(match)
    
    # 유사도 높은 순 정렬
    unique_matches.sort(key=lambda x: x['similarity'], reverse=True)
    return unique_matches[:top_n]


# 🆕 추가 함수: FAQ 직접 검색 (정확도 향상)
# ===================================================
def find_exact_faq_match(user_input):
    """완전 일치 또는 핵심 키워드 일치 FAQ 찾기"""
    user_clean = user_input.lower().replace(" ", "")
    
    # 핵심 단어 추출 (조사 제거)
    user_core = re.sub(r'(은|는|이|가|을|를|의|에|에서|으로|부터|까지|와|과)', '', user_clean)
    
    for faq in FAQ_DATA:
        faq_question = faq['질문'].lower().replace(" ", "")
        faq_core = re.sub(r'(은|는|이|가|을|를|의|에|에서|으로|부터|까지|와|과)', '', faq_question)
        
        # 1) 완전 일치
        if user_clean == faq_question or user_core == faq_core:
            return faq, 1.0
        
        # 2) 핵심 단어 포함 (80% 이상 일치)
        if len(user_core) >= 4 and len(faq_core) >= 4:
            if user_core in faq_core or faq_core in user_core:
                # 길이 비율 계산
                ratio = min(len(user_core), len(faq_core)) / max(len(user_core), len(faq_core))
                if ratio >= 0.6:
                    return faq, 0.9
    
    return None, 0


# === [핵심] Gemini API 답변 생성 ===
def generate_ai_response(user_input, chat_history, data_dict):
    """Gemini API를 사용하여 답변 생성"""
    
    # 1. 엑셀에서 관련 지식 추출
    context = get_ai_context(user_input, data_dict)


    # 2. FAQ 가져오기
    similar_faqs = get_faq_smart_search(user_input, top_n=3)
    
    faq_context = ""
    if similar_faqs:
        faq_context = "\n⭐⭐⭐ 다음 FAQ 정보를 반드시 활용하세요 ⭐⭐⭐\n\n"
        for idx, faq_item in enumerate(similar_faqs, 1):
            faq_context += f"━━━ FAQ {idx} (신뢰도: {faq_item['similarity']*100:.0f}%) ━━━\n"
            faq_context += f"질문: {faq_item['faq']['질문']}\n"
            faq_context += f"답변: {faq_item['faq']['답변']}\n\n"

    # 2-1. 정확한 매칭 먼저 시도
    exact_faq, exact_score = find_exact_faq_match(user_input)
    
    if exact_faq and exact_score >= 0.9:
        # 완전 일치 FAQ가 있으면 바로 반환
        faq_context = "\n🎯 **정확히 일치하는 FAQ를 찾았습니다!** 🎯\n\n"
        faq_context += f"질문: {exact_faq['질문']}\n"
        faq_context += f"답변: {exact_faq['답변']}\n\n"
        faq_context += "⚠️ 위 정보를 최우선으로 활용하여 답변하세요.\n"
    else:
        # 2-2. 유사 FAQ 검색
        similar_faqs = get_faq_smart_search(user_input, top_n=3)
        
        faq_context = ""
        if similar_faqs:
            faq_context = "\n⭐⭐⭐ 다음 FAQ 정보를 반드시 활용하세요 ⭐⭐⭐\n\n"
            for idx, faq_item in enumerate(similar_faqs, 1):
                faq_context += f"━━━ FAQ {idx} (신뢰도: {faq_item['similarity']*100:.0f}%) ━━━\n"
                faq_context += f"질문: {faq_item['faq']['질문']}\n"
                faq_context += f"답변: {faq_item['faq']['답변']}\n\n"
        
    
    # 3. 대화 기록
    history_text = ""
    for chat in chat_history[-2:]:
        history_text += f"{chat['role']}: {chat['content'][:80]}...\n"      

    # 4. ✅ 개선된 프롬프트 (
    prompt = f"""당신은 한경국립대학교 다전공 안내 전문 AI 상담원입니다.

질문에 답할 때, 아래 제공된 [학사 데이터]와 [FAQ 데이터]를 **동등하게 참고**하세요.


아래 [지침]을 철저히 준수하여 학생의 질문에 답변하세요.

[1. 답변 태도 및 형식]
- **톤앤매너**: 학생을 대하듯 따뜻하고 친절한 '습니다'체를 사용하세요.
- **적극성**: "홈페이지를 확인하세요", "자료가 부족합니다"라며 회피하지 마세요. 질문이 포괄적이라도 데이터에 있는 정보(전공 리스트, 연락처, 일반적인 신청 기간 등)를 최대한 활용하여 맛보기 정보를 먼저 제공해야 합니다.
- **시각화**: 
  - 중요한 수치(학점, 날짜 등)는 반드시 **굵게(**)** 표시하세요.
  - 연락처 안내, 학점 비교 시에는 **표(Table)**를 사용하여 깔끔하게 정리하세요.
  - 이모지(🎓, 📅, ✅)를 적절히 사용하여 딱딱하지 않게 하세요.
- **마무리**: 
  - 답변 끝에는 항상 연관 키워드(예: #복수전공 #신청기간)를 2~3개 달아주세요.
  - 추가 정보 확인을 위해 왼쪽 메뉴의 '다전공 제도 안내'를 참고하라고 안내하세요.
  - "💡 더 정확한 정보를 원하시면 '경영학전공 2학년 과목 알려줘'와 같이 질문해 주세요!"라는 가이드 문구를 포함하세요.   

[2. 데이터 활용 및 예외 처리]
- **데이터 우선**: 반드시 제공된 [학사 데이터]와 [FAQ]를 최우선으로 참고하세요.
- **데이터 부족 시**:
  - 일반적인 규정(예: 다전공 신청은 보통 4월/10월)을 먼저 안내하고,
  - "정확한 확인은 학사지원팀(031-670-5035) 또는 해당 전공 사무실 문의가 필요합니다"라고 덧붙이세요.
- **전공명 모호**: "행정"처럼 모호한 입력 시, "혹시 '행정학전공'을 찾으시나요?"라고 되묻고 데이터에 있는 유사 전공 리스트를 먼저 보여주세요.

[3. 학점 안내 (핵심)]
- **본전공 학점 변동 강조**: 다전공(복수전공, 부전공, 융합전공, 융합부전공) 신청 시 본전공 이수 학점이 줄어든다면 이를 강력히 어필하세요.
  - (예: "단일전공 시 **72학점**을 들어야 하지만, 복수전공 시 **36학점**으로 줄어듭니다! 표로 비교해 드릴게요.")
- **비교 설명**: 데이터의 '구분: 단일전공'과 '구분: 복수전공' 수치를 찾아 명확히 비교하세요. 본전공을 모를 경우 먼저 물어보세요.

[4. 마이크로디그리(MD) 특수 규정]
- 마이크로디그리 관련 문의할 때만  **학점 감면 없음**: "마이크로디그리는 본전공 학점 감면 혜택이 없으며, 기존 본전공 학점을 모두 이수해야 합니다"라고 명확히 경고하세요. (단, 과목 중복 인정은 가능함)
- 마이크로디그리 리스트를 안내할 때는 분야별로 구분해서 제시하세요.
- **문의처 안내**: 
  - 데이터에 **'교육운영전공'**이 있다면, "이 과정은 [000전공]에서 주관하므로 해당 사무실로 문의해주세요"라고 안내하고 전공 연락처, 위치를 알려주세요.
  - 정보가 없다면 "개설된 주관 전공 사무실로 문의해주세요"라고 안내하세요.

[5. 신청 절차 및 기간]
- **신청 자격 요건** 질문 시: 제공된 비교표를 활용하여 모든 제도의 신청 자격을 표 형식으로 깔끔하게 정리해서 보여주세요.
- 신청 관련 질문 시 복수/부전공/융합 등 유형별 **신청 기간(4월, 10월 등), 방법, 절차**를 데이터/FAQ에 기반해 구체적으로 안내하세요.
- ⭐ **포기/취소 관련 질문 시**: "포기는 별도의 신청 기간 없이 가능합니다"라고 먼저 안내한 후, FAQ 내용을 참고하도록 하세요.
- 절차 정보가 데이터에 없다면 "공통적으로 매 학기별(1학기는 4월/6월, 2학기는 9월/12월경) 2회씩 진행됩니다"라고 아는 범위 내에서 답변하세요.

[6. 과목 및 연락처 안내]
- **과목 추천**: 데이터에 있는 과목명을 구체적으로 언급하며 추천 이유를 짧게 덧붙이세요. 과목 리스트는 왼쪽 다전공 제도 안내를 참고하라고 해주세요.
- **연락처**: 표 형식으로 보여주되, 리스트가 길 경우 "더 궁금한 특정 전공이 있다면 말씀해 주세요"라고 자연스럽게 유도하세요.

[학사 데이터] 
{context[:4000] if context else "없음"}

[FAQ 데이터]
{faq_context if faq_context else "❌ 관련 FAQ 없음 - 이 경우에만 [학사 데이터] 활용"}

질문: {user_input}

답변 (중요 정보는 **강조**, 끝에 #키워드):"""


    # 5. API 호출    
    try:
        # 최신 google-genai SDK 호출 방식
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        if response and response.text:
            return response.text, "ai_generated"
        else:
            return "죄송합니다. 답변을 생성하지 못했습니다.", "error"
    except Exception as e:
        return f"AI 연결 오류가 발생했습니다: {str(e)}", "error"


    
# === 메인 화면 로직 수정 ===
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "안녕하세요! 한경국립대학교 다전공 안내 AI 비서입니다. 궁금한 점을 물어보세요! 🎓", "response_type": "greeting"}
    ]

def ask_chatbot(user_input):
    # 1. 키워드 기반으로 데이터에서 '지식 소스' 확보 (기존 keyword_match 로직 활용)
    # 직접 답변을 출력하지 않고, AI에게 전달할 context 변수에 담습니다.
    extracted_context = get_ai_context(user_input, all_data)
    
    # 2. 만약 정말 특수한 시스템 명령(예: "계산기 켜줘")이라면 즉시 처리
    if "계산기" in user_input:
        return "계산기 기능을 실행합니다.", "command_calc"

    # 3. 확보된 지식을 AI에게 던져서 "이 내용 기반으로 친절하게 답해줘"라고 시킴
    try:
        response_text, res_type = generate_ai_response(
            user_input,
            st.session_state.chat_history[:-1],
            ALL_DATA,            
        )
        return response_text, res_type
    except:
        # AI가 실패할 경우에만 백업용으로 기존 키워드 답변 출력 (Fallback)
        return generate_response(user_input)

# === 키워드 검색 함수 ===
def search_by_keyword(user_input):
    """키워드 기반 검색 (최우선)"""
    user_input_lower = user_input.lower()
    
    matched_keywords = []
    
    for keyword_data in KEYWORDS_DATA:
        keyword = keyword_data['키워드'].lower()
        
        if keyword in user_input_lower:
            matched_keywords.append(keyword_data)
    
    if matched_keywords:
        matched_keywords.sort(key=lambda x: len(x['키워드']), reverse=True)
        return matched_keywords[0]
    
    return None

def find_majors_with_details(user_input):
    """
    단어만 입력해도 전공명/키워드와 매칭하여 상세 정보를 반환
    """
    if MAJORS_INFO.empty:
        return []
    
    # 1. 입력값 정제 (공백 제거)
    user_input_clean = user_input.replace(" ", "").lower()
    
    # 입력값이 너무 짧으면(1글자) 검색 품질을 위해 제외 (예: '학', '과' 등)
    if len(user_input_clean) < 2:
        return []

    results = []
    
    for _, row in MAJORS_INFO.iterrows():
        # [수정] 마이크로디그리/소단위전공은 연락처 안내에서 제외
        p_type = str(row.get('제도유형', ''))
        if '마이크로' in p_type or '소단위' in p_type:
            continue

        # 데이터 정제
        major_name = str(row['전공명']).strip()
        major_clean = major_name.replace(" ", "").lower()
        
        # '전공', '학과', '학부'를 뗀 핵심 단어 추출 (예: 경영학전공 -> 경영학)
        core_name = major_clean.replace("전공", "").replace("학과", "").replace("학부", "")
        
        # 키워드 가져오기
        keywords = str(row.get('관심분야키워드', '')).lower()
        keyword_list = [k.strip().replace(" ", "") for k in keywords.split(',')]
        
        # === 매칭 로직 ===
        match_found = False
        priority = 0
        
        # Case A: 전공명에 입력어가 포함됨 (예: 입력 '경영' -> 데이터 '경영전공')
        if user_input_clean in major_clean: 
            match_found = True
            priority = 3  # 가장 높은 우선순위
            
        # Case B: 핵심 단어가 입력어와 같음 (예: 입력 '경영' -> 데이터 '경영학'의 핵심 '경영')
        elif core_name in user_input_clean:
            match_found = True
            priority = 2
            
        # Case C: 키워드 매칭 (예: 입력 '회계' -> 키워드 '회계')
        elif any(user_input_clean in k for k in keyword_list if k):
            match_found = True
            priority = 1

        if match_found:
            results.append({
                'major': major_name,
                'description': row.get('전공설명', '설명 없음'),
                'contact': row.get('연락처', '-'),
                'homepage': row.get('홈페이지', '-'),
                'location': row.get('위치', '-'),
                'program_types': row.get('제도유형', '-'),
                'priority': priority
            })
    
    # 우선순위 높음 -> 이름 짧은 순(정확도 높을 확률)으로 정렬
    results.sort(key=lambda x: (-x['priority'], len(x['major'])))
    
    return results


# === 유사도 기반 검색 함수 ===
@st.cache_resource
def create_faq_vectorizer():
    """FAQ 질문들을 벡터화"""
    questions = [faq['질문'] for faq in FAQ_DATA]
    vectorizer = TfidfVectorizer()
    
    if questions:
        vectors = vectorizer.fit_transform(questions)
        return vectorizer, vectors, questions
    return None, None, []

def find_similar_faq(user_input, threshold=0.3):
    """유사한 FAQ 찾기"""
    vectorizer, faq_vectors, questions = create_faq_vectorizer(force_reload=True)
    
    if vectorizer is None or not questions:
        return None
    
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    max_idx = np.argmax(similarities)
    if similarities[max_idx] >= threshold:
        return FAQ_DATA[max_idx], similarities[max_idx]
    return None

def get_top_similar_faqs(user_input, top_n=3):
    """가장 유사한 FAQ 여러 개 반환"""
    vectorizer, faq_vectors, questions = create_faq_vectorizer()
    
    if vectorizer is None or not questions:
        return []
    
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.05:
            results.append({
                'faq': FAQ_DATA[idx],
                'similarity': similarities[idx]
            })
    
    return results

def find_similar_program(user_input):
    """제도명 유사도 검색"""
    program_names = list(PROGRAM_INFO.keys())
    
    for program in program_names:
        if program in user_input:
            return program
    
    for program in program_names:
        if any(word in user_input for word in program.split()):
            return program
    
    return None


# === 🆕 관심분야 기반 전공 추천 함수 ===
def recommend_majors_by_interest(user_input):
    """관심분야 키워드 매칭 로직 개선"""
    # 1. 데이터 로드 확인
    if MAJORS_INFO.empty:
        return []
    
    # 2. 필수 컬럼 확인 (컬럼명이 다를 경우를 대비해 유연하게 처리 가능)
    if '관심분야키워드' not in MAJORS_INFO.columns:
        # 컬럼명이 다를 경우 수동으로 매핑하거나 빈 리스트 반환
        return []

    user_input_lower = user_input.lower()
    recommendations = []
    
    for _, row in MAJORS_INFO.iterrows():
        # 데이터 전처리 (NaN 처리 및 문자열 변환)
        raw_keywords = str(row.get('관심분야키워드', ''))
        if raw_keywords == 'nan' or not raw_keywords.strip():
            continue
            
        # 콤마(,) 기준으로 나누고 공백 제거
        keywords_list = [k.strip().lower() for k in raw_keywords.split(',')]
        
        # 3. 매칭 검사: 입력 문장에 키워드가 포함되어 있는지 확인
        # (예: 입력 "인공지능 배우고 싶어" -> 키워드 "인공지능" 매칭)
        matched = [k for k in keywords_list if k in user_input_lower]
        
        if matched:
            recommendations.append({
                'major': row['전공명'],
                'description': row.get('전공설명', '설명 없음'),
                'program_types': row.get('제도유형', '-'),
                'match_score': len(matched), # 매칭된 키워드 개수로 점수 산정
                'matched_keywords': matched,
                'contact': row.get('연락처', '-'),
                'homepage': row.get('홈페이지', '-')
            })
    
    # 매칭 점수가 높은 순으로 정렬 후 상위 5개 반환
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return recommendations[:5]

def display_major_info(major_name):
    """특정 전공의 연락처/홈페이지 정보 표시"""
    if MAJORS_INFO.empty:
        return "전공 정보를 불러올 수 없습니다."
    
    major_data = MAJORS_INFO[MAJORS_INFO['전공명'] == major_name]
    
    if major_data.empty:
        return f"'{major_name}' 전공 정보를 찾을 수 없습니다."
    
    row = major_data.iloc[0]
    
    response = f"**{major_name} 📞**\n\n"
    response += f"**📝 소개:** {row['전공설명']}\n\n"
    response += f"**📚 이수 가능 다전공 제도:** {row['제도유형']}\n\n"
    response += f"**📞 연락처:** {row['연락처']}\n\n"
    
    if pd.notna(row.get('홈페이지')) and row['홈페이지'] != '-':
        response += f"**🌐 홈페이지:** {row['홈페이지']}\n\n"
    
    if pd.notna(row.get('위치')) and row['위치'] != '-':
        response += f"**📍 위치:** {row['위치']}\n\n"
    
    return response


# === 이미지 표시 함수 ===
def display_curriculum_image(major, program_type):
    """이수체계도 또는 안내 이미지 표시"""
    result = CURRICULUM_MAPPING[
        (CURRICULUM_MAPPING['전공명'] == major) & 
        (CURRICULUM_MAPPING['제도유형'] == program_type)
    ]
    
    if not result.empty:
        raw_filenames = str(result.iloc[0]['파일명'])
        filenames = [f.strip() for f in raw_filenames.split(',')]
        
        if len(filenames) > 1:
            cols = st.columns(len(filenames)) 
            for idx, filename in enumerate(filenames):
                image_path = f"images/curriculum/{filename}"
                with cols[idx]:
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"{major} 안내-{idx+1}", use_container_width=True)
                    else:
                        st.warning(f"⚠️ 이미지 파일 없음: {filename}")
            return True
            
        else:
            filename = filenames[0]
            image_path = f"images/curriculum/{filename}"
            
            if os.path.exists(image_path):
                is_micro = "소단위전공과정(마이크로디그리)" in program_type or "마이크로디그" in program_type
                caption_text = f"{major} 안내 이미지" if is_micro else f"{major} 이수체계도"
                
                if is_micro:
                    col1, col2, col3 = st.columns([1, 2, 1]) 
                    with col2:
                        st.image(image_path, caption=caption_text, use_container_width=True)
                else:
                    st.image(image_path, caption=caption_text, use_container_width=True)
                
                return True
            else:
                st.warning(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")
                return False
    else:
        if "소단위전공과정(마이크로디그리)" not in program_type:
            st.info(f"💡 {major} {program_type}의 이수체계도가 준비 중입니다.")
        return False
    
# === 과목 표시 함수 ===
def display_courses(major, program_type):
    """과목 정보 표시"""
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == major) & 
        (COURSES_DATA['제도유형'] == program_type)
    ]
    
    if not courses.empty:
        st.subheader(f"📚 {major} 편성 교과목(2025학년도 교육과정)")       
        
        if "소단위전공과정(마이크로디그리)" in program_type:
            semesters = sorted(courses['학기'].unique())
            
            for semester in semesters:
                st.markdown(f"#### {int(semester)}학기")
                
                semester_courses = courses[courses['학기'] == semester]
                
                for _, course in semester_courses.iterrows():
                    division = course['이수구분']
                    course_name = course['과목명']
                    credits = int(course['학점'])
                    
                    if division in ['전필', '필수']:
                        badge_color = "🔴"
                    elif division in ['전선', '선택']:
                        badge_color = "🟢"
                    else:
                        badge_color = "🔵"
                    
                    st.write(f"{badge_color} **[{division}]** {course_name} ({credits}학점)")
                
                st.write("")
                
        else:
            years = sorted([int(y) for y in courses['학년'].unique() if pd.notna(y)])
            
            if len(years) > 0:
                tabs = st.tabs([f"{year}학년" for year in years])
                
                for idx, year in enumerate(years):
                    with tabs[idx]:
                        year_courses = courses[courses['학년'] == year]
                        semesters = sorted(year_courses['학기'].unique())
                        
                        for semester in semesters:
                            st.write(f"**{int(semester)}학기**")
                            semester_courses = year_courses[year_courses['학기'] == semester]
                            
                            for _, course in semester_courses.iterrows():
                                division = course['이수구분']
                                course_name = course['과목명']
                                credits = int(course['학점'])
                                
                                if division in ['전필', '필수']:
                                    badge_color = "🔴"
                                elif division in ['전선', '선택']:
                                    badge_color = "🟢"
                                else:
                                    badge_color = "🔵"
                                
                                st.write(f"{badge_color} **[{division}]** {course_name} ({credits}학점)")
                            
                            st.write("")
               
        return True
    else:
        return False

# === 비교표 생성 ===
def create_comparison_table():
    data = {
        "제도": list(PROGRAM_INFO.keys()),
        "이수학점(교양)": [info["credits_general"] for info in PROGRAM_INFO.values()],
        "원전공 이수학점": [info["credits_primary"] for info in PROGRAM_INFO.values()],
        "다전공 이수학점": [info["credits_multi"] for info in PROGRAM_INFO.values()],
        "졸업인증": [info["graduation_certification"] for info in PROGRAM_INFO.values()],
        "졸업시험": [info["graduation_exam"] for info in PROGRAM_INFO.values()],
        "학위기 표기": [info["degree"] for info in PROGRAM_INFO.values()],
        "난이도": [info["difficulty"] for info in PROGRAM_INFO.values()],
        "신청자격": [info["qualification"] for info in PROGRAM_INFO.values()]
    }
    return pd.DataFrame(data)

# === 챗봇 응답 생성 ===
def generate_response(user_input):
    user_input_lower = user_input.lower()
    
    # 1. 인사
    if any(x in user_input_lower for x in ["안녕", "하이", "hello", "반가"]):
        return "안녕하세요! 👋 유연학사제도(다전공) 안내 AI챗봇입니다. 궁금한 전공이나 제도를 물어보세요!", "greeting"

    # ====================================================
    # 2. [통합 검색] 전공/관심분야 검색 (최우선 처리)
    # "경영", "컴퓨터 연락처", "AI 추천" 등 모든 케이스를 여기서 처리
    # ====================================================
    search_results = find_majors_with_details(user_input)
    
    if search_results:
        response = f"**🔍 '{user_input}' 관련 전공 정보입니다.**\n\n"
        
        # 상위 3개만 표시
        for idx, info in enumerate(search_results[:3], 1):
            response += f"### {idx}. {info['major']}\n"
            
            # 소개 (설명이 없으면 생략)
            if info['description'] and info['description'] != '설명 없음':
                response += f"**📝 소개:** {info['description']}\n\n"
            
            # 연락처 (필수 정보)
            response += f"**📞 연락처:** {info['contact']}\n"
            
            # 홈페이지 (정보가 있는 경우만 표시)
            if info['homepage'] not in ['-', 'nan', None, '']:
                 response += f"**🌐 홈페이지:** [{info['homepage']}]({info['homepage']})\n"
            
            # 위치 (정보가 있는 경우만 표시)
            if info['location'] not in ['-', 'nan', None, '']:
                response += f"**📍 전공 사무실 위치:** {info['location']}\n"
            
            # 제도 유형
            response += f"\n**🎓 이수 가능 다전공:** {info['program_types']}\n"
            response += "\n"
            
        return response, "major_info"

    # ====================================================
    # 3. [예외 처리] 전공명 없이 '연락처'만 물어본 경우
    # 검색 결과가 없을 때만 실행됨 -> 전체 목록 제공
    # ====================================================
    if any(word in user_input_lower for word in ["연락처", "전화번호", "과사", "사무실"]):
        response = "**📞 전공별 연락처 안내**\n\n"
        response += "찾으시는 **전공명을 정확히 말씀해주시면** 해당 사무실 정보를 안내해드립니다.\n"
        response += "아래 목록에 있는 전공명을 입력해 보세요.\n\n"
        
        if not MAJORS_INFO.empty:
            # 1. 데이터 정리
            df_clean = MAJORS_INFO.dropna(subset=['전공명']).copy()
            df_clean['전공명'] = df_clean['전공명'].astype(str)
            
            # 2. 그룹 분리 로직 (마이크로디그리 vs 일반)
            try:
                is_md = df_clean['제도유형'].str.contains('마이크로|소단위', na=False) | \
                        df_clean['전공명'].str.contains('마이크로|소단위', na=False)
            except KeyError:
                is_md = df_clean['전공명'].str.contains('마이크로|소단위', na=False)

            general_majors = sorted(df_clean[~is_md]['전공명'].unique())
            md_majors = sorted(df_clean[is_md]['전공명'].unique())
            
            # 3. 일반 전공 출력
            response += "### 🏫 학부/전공\n"
            if general_majors:
                for i in range(0, len(general_majors), 3):
                    batch = general_majors[i:i+3]
                    response += " | ".join(batch) + "\n"
            
            # 4. 마이크로디그리 출력
            if md_majors:
                response += "\n### 🎓 소단위전공(마이크로디그리)\n"
                for i in range(0, len(md_majors), 2):
                    batch = md_majors[i:i+2]
                    response += " | ".join(batch) + "\n"
        
        return response, "contact_list"

    # ====================================================
    # 4. 제도 키워드 검색
    # ====================================================
    keyword_match = search_by_keyword(user_input)
    if keyword_match:
        keyword_type = keyword_match['타입']
        linked_info = keyword_match['연결정보']
        
        if keyword_type == "제도" and linked_info in PROGRAM_INFO:
            info = PROGRAM_INFO[linked_info]
            response = f"**{linked_info}** 📚\n\n"
            response += f"**설명:** {info['description']}\n\n"
            response += f"**📖 이수학점**\n"
            response += f"- 교양: {info['credits_general']}\n"
            response += f"- 원전공: {info['credits_primary']}\n\n"
            response += f"- 다전공: {info['credits_multi']}\n\n"
            response += f"**🎓 졸업 요건**\n"
            response += f"- 졸업인증: {info['graduation_certification']}\n"
            response += f"- 졸업시험: {info['graduation_exam']}\n\n"
            response += f"**✅ 신청자격:** {info['qualification']}\n"
            response += f"**📜 학위기 표기:** {info['degree']}\n"
            response += f"**♧ 난이도:** {info['difficulty']}\n\n"
            
            if info['features']:
                response += f"**✨ 특징:**\n"
                for feature in info['features']:
                    response += f"- {feature.strip()}\n"
            if info['notes']:
                response += f"\n**💡 기타:** {info['notes']}"
                
            response += f"\n\n_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
            return response, "program" # [수정] 올바른 response 리턴
        
        elif keyword_type == "주제":
            if linked_info == "학점정보":
                response = "**제도별 이수 학점** 📖\n\n"
                for program, info in PROGRAM_INFO.items():
                    response += f"**{program}**\n"
                    response += f"  - 교양: {info['credits_general']}\n"
                    response += f"  - 원전공: {info['credits_primary']}\n\n"
                    response += f"  - 다전공: {info['credits_multi']}\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "credits"
            
            elif linked_info == "신청정보":
                response = "**신청 관련 정보** 📝\n\n"
                response += "다전공 제도는 매 학기 초(4월, 10월), 학기말(6월, 12월)에 신청 가능합니다.\n\n"
                response += "자세한 내용은 '📚 다전공 제도 안내' 또는 '❓ FAQ' 메뉴'를 확인하시거나, - [📥 홈페이지 학사공지](https://www.hknu.ac.kr/kor/562/subview.do)\n를 참고해 주세요!\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "application"
            
            elif linked_info == "비교표":
                response = "각 제도의 비교는 왼쪽 사이드바의 '📚 다전공 제도 안내'에서 확인하실 수 있습니다!\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "comparison"
            
            elif linked_info == "졸업요건":
                response = "**제도별 졸업 요건** 🎓\n\n"
                for program, info in PROGRAM_INFO.items():
                    response += f"**{program}**\n"
                    response += f"  - 졸업인증: {info['graduation_certification']}\n"
                    response += f"  - 졸업시험: {info['graduation_exam']}\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "graduation"
    
    # ====================================================
    # 5. FAQ 및 기타 로직
    # ====================================================
    
    # FAQ 유사도 검색
    similar_faq = find_similar_faq(user_input)
    if similar_faq:
        faq, similarity = similar_faq
        response = f"**Q. {faq['질문']}**\n\nA. {faq['답변']}\n\n"
        response += f"_💡 답변 신뢰도: {similarity*100:.0f}%_"
        return response, "faq"
    
    # 제도 설명 검색 (유사도)
    program = find_similar_program(user_input)
    if program:
        info = PROGRAM_INFO[program]
        response = f"**{program}** 📚\n\n"
        response += f"**설명:** {info['description']}\n..." # (길어서 생략, 필요한 경우 위와 동일하게 작성)
        return response, "program"
    
    # 비교 질문
    if any(word in user_input_lower for word in ["비교", "차이", "다른점", "vs"]):
        return "각 제도의 비교는 왼쪽 사이드바의 '📚 다전공 제도 안내'에서 확인하실 수 있습니다!", "comparison"
    
    # 학점 관련 (키워드 매칭 실패 시 백업)
    if any(word in user_input_lower for word in ["학점", "몇학점"]):
        response = "**제도별 이수 학점** 📖\n\n"
        for program, info in PROGRAM_INFO.items():
            response += f"**{program}**\n - 교양: {info['credits_general']}\n - 원전공: {info['credits_primary']}\n - 다전공: {info['credits_multi']}\n\n"
        return response, "credits"
    
    # 신청 관련 (백업)
    if any(word in user_input_lower for word in ["신청", "지원", "언제", "기간"]):
        return "매 학기 초(4월, 10월) 및 학기말(6월, 12월)에 신청 가능합니다.", "application"
    
    # 유사 질문 제안
    similar_faqs = get_top_similar_faqs(user_input, top_n=3)
    if similar_faqs:
        response = "정확히 일치하는 답변을 찾지 못했습니다. 😅\n\n**혹시 다음 질문 중 하나를 찾으셨나요?**\n\n"
        for i, item in enumerate(similar_faqs, 1):
            response += f"{i}. {item['faq']['질문']} _({item['similarity']*100:.0f}%)_\n"
        return response, "suggestion"
    
    # 완전 매칭 실패
    return "죄송합니다. 질문을 이해하지 못했습니다. 😅\n'경영'이나 '복수전공'처럼 핵심 단어로 질문해 보시겠어요?", "no_match"

# === 메인 UI ===
def main():
    # 1. 데이터 로드
    ALL_DATA = load_all_data()
    courses_data = load_courses()
    majors_info = load_majors_info()

    st.title("🎓 한경국립대 유연학사제도(다전공) 안내")
    
    # === 사이드바 설정 ===
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 10px 0;'>
                <h1 style='font-size: 3rem; margin-bottom: 0;'>🎓</h1>
                <h3 style='margin-top: 0;'>HKNU 다전공 제도 안내</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        menu = option_menu(
            menu_title=None,
            options=["AI챗봇 상담", "다전공 제도 안내", "FAQ"], 
            icons=["chat-dots-fill", "journal-bookmark-fill", "question-circle-fill"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#0091FF"},
            }
        )

        st.divider()

        with st.container(border=True):
            st.markdown("### 🤖 다전공 안내 AI챗봇")
            st.info(
                """
                **AI챗봇이 여러분의 다전공 고민을
                해결해 드립니다.
                
                *"경영학과 졸업요건은?"*
                *"복수전공 신청 기간은?"*
                
                무엇이든 물어보세요!
                """
            )

            st.caption("* 본 챗봇에서 제공하는 정보는 단순 참고용입니다.")

        with st.container(border=True):
            st.markdown("### 📚 다전공 제도란?")
            st.success("주전공 외에 복수/부전공, 융합/융합부전공, 마이크로디그리 등 다양한 전공을 이수할 수 있는 제도입니다.")
            
        st.markdown("---")
        st.caption("☏ 학사지원팀 031-670-5035")
        st.caption("* Powered by Gemini 2.0")


    # === 메인 콘텐츠 영역 ===
    
    if menu == "AI챗봇 상담":
        st.subheader("💬 AI 상담원과 대화하기")

        # [복원됨] 👋 상단 질문 예시 가이드
        with st.expander("💡 어떤 질문을 해야 할지 모르겠나요? (클릭)", expanded=False):
            st.markdown("아래 탭을 눌러 관심 있는 주제의 질문을 선택해보세요!")
            
            
            # 탭 생성
            tab1, tab2, tab3, tab4 = st.tabs(["🔰 기초/제도", "📅 신청/절차", "🎓 학점/졸업", "🏫 전공/과목"])

            # 질문 데이터를 딕셔너리로 관리
            questions = {
                "tab1": [
                    "복수전공과 부전공의 차이점은?",
                    "융합전공에는 어떤 것들이 있어?",
                    "소단위전공과정(마이크로디그리)가 뭐야?",
                    "연계전공은 무엇인가요?",
                    "다전공을 꼭 해야 하나요?"
                ],
                "tab2": [
                    "다전공 신청 기간은 언제야?",
                    "복수전공 신청 방법 알려줘",
                    "신청 자격 요건이 어떻게 돼?",
                    "다전공 포기(취소)는 언제 할 수 있어?",
                    "융합전공 신청 절차가 궁금해"
                ],
                "tab3": [
                    "복수전공하면 졸업학점 어떻게 변해?",
                    "부전공 이수 학점 알려줘",
                    "마이크로디그리 들으면 본전공 학점 줄어들어?",
                ],
                "tab4": [
                    "경영학전공 사무실 전화번호 알려줘",
                    "소프트웨어융합전공 2학년 과목 보여줘",
                    "식품품질관리 마이크로디그리 과목 리스트",
                    "시각미디어디자인전공 사무실 위치 어디야?",
                ]
            }

        
        # 헬퍼 함수: 버튼 생성 및 클릭 처리
        def create_question_buttons(tab_name, q_list):
            with tab_name:
                # 2단 그리드로 배치
                cols = st.columns(2)
                for idx, q in enumerate(q_list):
                    if cols[idx % 2].button(f"💬 {q}", key=f"{tab_name}_{idx}", use_container_width=True):
                        # 클릭 시 동작
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        with st.spinner("AI가 답변을 생성 중입니다..."):
                            response_text, res_type = generate_ai_response(
                                q,
                                st.session_state.chat_history[:-1],
                                ALL_DATA
                            )
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": response_text, "response_type": res_type
                        })
                        st.rerun()

        # 각 탭에 버튼 그리기
        create_question_buttons(tab1, questions["tab1"])
        create_question_buttons(tab2, questions["tab2"])
        create_question_buttons(tab3, questions["tab3"])
        create_question_buttons(tab4, questions["tab4"])
        

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
                    
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "response_type": res_type})
            scroll_to_bottom()

    # === [화면 2] 다전공 제도 안내 ===
    elif menu == "다전공 제도 안내":
        st.header("📊 제도 한눈에 비교")

        # 1. 상단 카드형 UI
        if 'programs' in ALL_DATA and ALL_DATA['programs']:
            cols = st.columns(3)
            for idx, (program, info) in enumerate(ALL_DATA['programs'].items()):
                with cols[idx % 3]:
                    desc = info.get('description', '설명 없음')
                    c_pri = info.get('credits_primary', '-')
                    c_mul = info.get('credits_multi', '-')
                    degree = info.get('degree', '-')
                    difficulty = info.get('difficulty', '⭐')
                    
                    long_text_style = "overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.4;"

                    html_content = f"""
                    <div style="border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); min-height: 380px; margin-bottom: 20px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <h3 style="margin: 0 0 8px 0; color: #1f2937; font-size: 1.2rem;">🎓 {program}</h3>
                            <p style="color: #6b7280; font-size: 14px; margin-bottom: 12px; {long_text_style}">{desc}</p>
                            <hr style="margin: 12px 0; border: 0; border-top: 1px solid #e5e7eb;">
                            <div style="font-size: 14px; margin-bottom: 8px;">
                                <strong style="color: #374151;">📖 이수 학점</strong>
                                <ul style="padding-left: 18px; margin: 4px 0; color: #4b5563;">
                                    <li style="margin-bottom: 4px;"><span style="font-weight:600; color:#374151;">본전공:</span> {c_pri}</li>
                                    <li><span style="font-weight:600; color:#374151;">다전공:</span> {c_mul}</li>
                                </ul>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: end; margin-top: 10px;">
                            <div style="max-width: 65%;">
                                <strong style="color: #374151; font-size: 14px;">📜 학위기</strong><br>
                                <div style="font-size: 13px; color: #2563eb; background: #eff6ff; padding: 2px 6px; border-radius: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{degree}</div>
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

        # 2. 상세 조회 기능
        st.subheader("🔍 상세 정보 조회")
        
        prog_keys = list(ALL_DATA['programs'].keys()) if 'programs' in ALL_DATA else []
        selected_program = st.selectbox("자세히 알아볼 제도를 선택하세요", prog_keys)
        
        if selected_program and 'programs' in ALL_DATA:
            info = ALL_DATA['programs'][selected_program]
            
            # 기본 정보 탭
            tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징 및 유의사항"])
            with tab1:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**개요**\n\n{info.get('description', '-')}")
                    st.subheader("📖 이수 학점 상세")
                    st.markdown(f"""
                    - **교양:** {info.get('credits_general', '-')}
                    - **원전공:** {info.get('credits_primary', '-')}
                    - **다전공:** {info.get('credits_multi', '-')}
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

            # [✨ 복원된 기능] 오리지널 이수 학점 확인 로직
            
            # 1) 전공 목록 확보
            available_majors = set()
            
            # courses 데이터에서 가져오기
            if 'courses' in ALL_DATA and not ALL_DATA['courses'].empty:
                c_df = ALL_DATA['courses']
                if '제도유형' in c_df.columns:
                    # 소단위전공은 다양한 키워드로 검색
                    if "소단위" in selected_program or "마이크로" in selected_program:
                        mask = c_df['제도유형'].astype(str).str.contains('소단위|마이크로|MD', case=False, na=False)
                    else:
                        mask = c_df['제도유형'].astype(str).str.contains(selected_program, na=False)
                    available_majors.update(c_df[mask]['전공명'].unique())

            # curriculum 데이터에서 가져오기
            if 'curriculum' in ALL_DATA:
                 curr_df = ALL_DATA['curriculum']
                 if not curr_df.empty and '제도유형' in curr_df.columns:
                     # 소단위전공은 다양한 키워드로 검색
                     if "소단위" in selected_program or "마이크로" in selected_program:
                         mask = curr_df['제도유형'].astype(str).str.contains('소단위|마이크로|MD', case=False, na=False)
                     else:
                         mask = curr_df['제도유형'].astype(str).str.contains(selected_program, na=False)
                     available_majors.update(curr_df[mask]['전공명'].unique())
            
            # 🆕 majors_info 데이터에서도 가져오기 (중요!)
            if 'majors' in ALL_DATA and not ALL_DATA['majors'].empty:
                m_df = ALL_DATA['majors']
                if '제도유형' in m_df.columns:
                    # 소단위전공은 다양한 키워드로 검색
                    if "소단위" in selected_program or "마이크로" in selected_program:
                        mask = m_df['제도유형'].astype(str).str.contains('소단위|마이크로|MD', case=False, na=False)
                    else:
                        mask = m_df['제도유형'].astype(str).str.contains(selected_program, na=False)
                    available_majors.update(m_df[mask]['전공명'].unique())

            # 2) 전공 선택 UI (오리지널 스타일)
            if available_majors:
                target_programs = ["복수전공", "부전공", "융합전공", "융합부전공"]
                
                if selected_program in target_programs:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors)))
                    with col_m2:
                        # 본전공 목록 불러오기
                        all_majors_list = []
                        if 'primary_req' in ALL_DATA and not ALL_DATA['primary_req'].empty:
                            all_majors_list = sorted(ALL_DATA['primary_req']['전공명'].unique().tolist())
                        my_primary_major = st.selectbox("나의 본전공 (제1전공)", ["선택 안 함"] + all_majors_list)
                else:
                    selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors)))
                    my_primary_major = "선택 안 함"

                # 3) 학점 요건 표시 (오리지널 로직)
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
                    
                    # 왼쪽: 타겟 전공 요건
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
                                
                                st.write(f"- 전공필수: **{int(row['전공필수'])}**학점")
                                st.write(f"- 전공선택: **{int(row['전공선택'])}**학점")
                                st.markdown(f"#### 👉 {selected_program} {int(row['총학점'])}학점")
                            else:
                                st.warning(f"{admission_year}학번 기준 데이터가 없습니다.")
                        else:
                            st.warning("졸업요건 데이터가 없습니다.")

                    # 오른쪽: 본전공 변동 요건
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
                                        if selected_program in str(p_row['구분']):
                                            matched_row = p_row
                                            break
                                    
                                    if matched_row is not None:
                                        st.write(f"- 본전공 전필: **{int(matched_row['본전공_전필'])}**학점")
                                        st.write(f"- 본전공 전선: **{int(matched_row['본전공_전선'])}**학점")
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

                # 이미지 표시
                if selected_program == "융합전공" or "소단위전공" in selected_program:
                    title = "📋 이수체계도" if selected_program == "융합전공" else "🖼️ 과정 안내 이미지"
                    st.subheader(title)
                    display_curriculum_image(selected_major, selected_program)
        
                # 이수 과목 표시
                if not COURSES_DATA.empty:
                    display_courses(selected_major, selected_program)

    # === [화면 3] FAQ ===
    elif menu == "FAQ":
        st.header("❓ 자주 묻는 질문")
        if 'faq' in ALL_DATA and ALL_DATA['faq']:
            for faq in ALL_DATA['faq']:
                with st.expander(f"Q. {faq['질문']}"):
                    st.write(f"A. {faq['답변']}")
        else:
            st.info("등록된 FAQ가 없습니다.")

if __name__ == "__main__":
    main()
