"""
============================================================
🎯 다전공 추천 시뮬레이션 모듈
============================================================
버전: 1.0
설명: 학생의 학점 현황을 분석하고 최적의 다전공 제도를 추천
============================================================
"""

import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import os

# ============================================================
# 상수 정의
# ============================================================

MAX_CREDITS_PER_SEMESTER = 18  # 학기당 최대 수강학점
FRESHMAN_TOTAL_SEMESTERS = 8   # 신입학 총 학기
TRANSFER_TOTAL_SEMESTERS = 4   # 편입학 총 학기
MAX_DOUBLE_MAJOR_CREDITS = 130  # 복수전공 최대 졸업학점
DEFAULT_GRADUATION_CREDITS = 120  # 기본 졸업학점

# 제도 우선순위 (숫자가 낮을수록 우선)
PROGRAM_PRIORITY = {
    "복수전공": 1,
    "융합전공": 2,
    "부전공": 3,
    "융합부전공": 4,
    "연계전공": 5,
}

# ============================================================
# 열거형 정의
# ============================================================

class AdmissionType(Enum):
    FRESHMAN = "신입학"
    TRANSFER_SAME = "3학년 편입학(동일계)"
    TRANSFER_DIFF = "3학년 편입학(비동일계)"

class StudentType(Enum):
    NEW_APPLICANT = "신규 신청자"
    CURRENT_PARTICIPANT = "기존 참여자"

class GraduationStatus(Enum):
    POSSIBLE = "가능"
    RISKY = "위험"
    IMPOSSIBLE = "어려움"


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class StudentInput:
    """학생 입력 정보"""
    # 기본 정보
    student_type: str                    # 신규 신청자 / 기존 참여자
    admission_year: int                  # 입학연도
    primary_major: str                   # 본전공명
    admission_type: str                  # 입학구분
    completed_semesters: int             # 현재까지 이수한 학기 수
    transfer_credits: int = 0            # 편입학 인정학점
    
    # 교양 학점 (신입학만)
    credits_basic_literacy: int = 0      # 기초교양(기초문해)
    credits_basic_science: int = 0       # 기초교양(기초과학)
    credits_core_liberal: int = 0        # 핵심교양
    
    # 본전공 학점
    credits_major_required: int = 0      # 전공필수 이수 학점
    credits_major_elective: int = 0      # 전공선택 이수 학점
    
    # 잔여 학점
    credits_free: int = 0                # 잔여(자유) 이수 학점
    
    # 다전공 관련 (신규 신청자)
    desired_multi_major: Optional[str] = None
    
    # 다전공 관련 (기존 참여자)
    current_program: Optional[str] = None
    current_multi_major: Optional[str] = None
    credits_multi_required: int = 0
    credits_multi_elective: int = 0


@dataclass
class CreditAnalysis:
    """학점 분석 결과"""
    # 기준 학점
    req_major_required: int = 0
    req_major_elective: int = 0
    req_multi_required: int = 0
    req_multi_elective: int = 0
    req_graduation_credits: int = 120
    
    # 이수 학점
    completed_major_required: int = 0
    completed_major_elective: int = 0
    completed_multi_required: int = 0
    completed_multi_elective: int = 0
    completed_total: int = 0
    
    # 부족 학점
    deficit_major_required: int = 0
    deficit_major_elective: int = 0
    deficit_multi_required: int = 0
    deficit_multi_elective: int = 0
    deficit_graduation: int = 0
    
    # 학기 정보
    remaining_semesters: int = 0
    max_additional_credits: int = 0
    
    # 본전공 변화 학점 (다전공 참여 시)
    req_major_required_changed: int = 0
    req_major_elective_changed: int = 0


@dataclass
class SimulationResult:
    """제도별 시뮬레이션 결과"""
    program_type: str                    # 제도 유형
    multi_major_name: str                # 다전공 전공명
    
    # 졸업 가능 여부
    can_graduate: bool = False
    graduation_status: str = "어려움"
    
    # 학점 분석
    credit_analysis: CreditAnalysis = field(default_factory=CreditAnalysis)
    
    # 추천 정보
    recommendation_rank: int = 0
    recommendation_reason: str = ""
    is_supplementary: bool = False
    
    # 학기별 이수 계획
    semester_plan: List[Dict] = field(default_factory=list)


@dataclass
class AnalysisOutput:
    """전체 분석 결과"""
    student_input: StudentInput = None
    
    # 현재 상태 분석
    current_analysis: CreditAnalysis = field(default_factory=CreditAnalysis)
    current_can_graduate: bool = False
    
    # 제도별 시뮬레이션 결과
    simulation_results: List[SimulationResult] = field(default_factory=list)
    
    # 추천 결과
    recommended_programs: List[SimulationResult] = field(default_factory=list)
    supplementary_programs: List[SimulationResult] = field(default_factory=list)


# ============================================================
# 데이터 로드 함수
# ============================================================

@st.cache_data
def load_primary_requirements():
    """본전공 기준 데이터 로드"""
    try:
        return pd.read_excel('data/primary_requirements.xlsx')
    except:
        return pd.DataFrame()

@st.cache_data
def load_graduation_requirements():
    """다전공 기준 데이터 로드"""
    try:
        return pd.read_excel('data/graduation_requirements.xlsx')
    except:
        return pd.DataFrame()

@st.cache_data
def load_majors_list():
    """전공 목록 로드"""
    try:
        majors_df = pd.read_excel('data/majors_info.xlsx')
        return sorted(majors_df['전공명'].unique().tolist())
    except:
        return []

@st.cache_data
def load_multi_majors_by_program(program_type: str):
    """제도별 다전공 목록 로드"""
    try:
        gr_df = pd.read_excel('data/graduation_requirements.xlsx')
        filtered = gr_df[gr_df['제도유형'] == program_type]
        return sorted(filtered['전공명'].unique().tolist())
    except:
        return []


def safe_int(value, default=0):
    """안전하게 정수로 변환"""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def get_primary_requirement(
    primary_major: str,
    program_type: str,
    admission_year: int,
    pr_df: pd.DataFrame
) -> Optional[Dict]:
    """본전공 기준 조회"""
    if pr_df.empty:
        return None
    
    # 정확한 매칭
    result = pr_df[
        (pr_df['전공명'] == primary_major) &
        (pr_df['제도유형'] == program_type) &
        (pr_df['기준학번'] == admission_year)
    ]
    
    if result.empty:
        # 부분 매칭 시도 (전공명에 키워드 포함)
        keyword = primary_major.replace('전공', '').replace('(평캠)', '').replace('(평택)', '').strip()
        if keyword:
            result = pr_df[
                (pr_df['전공명'].str.contains(keyword, case=False, na=False)) &
                (pr_df['제도유형'] == program_type) &
                (pr_df['기준학번'] == admission_year)
            ]
    
    if result.empty:
        # 가장 가까운 학번으로 대체
        result = pr_df[
            (pr_df['전공명'] == primary_major) &
            (pr_df['제도유형'] == program_type)
        ]
        if not result.empty:
            closest_year = min(result['기준학번'].unique(), key=lambda x: abs(x - admission_year))
            result = result[result['기준학번'] == closest_year]
    
    if result.empty:
        # 기본값 반환
        return {
            'major_name': primary_major,
            'program_type': program_type,
            'admission_year': admission_year,
            'req_major_required': 15,
            'req_major_elective': 33,
            'req_total': 48,
        }
    
    row = result.iloc[0]
    return {
        'major_name': row['전공명'],
        'program_type': row['제도유형'],
        'admission_year': safe_int(row['기준학번'], admission_year),
        'req_major_required': safe_int(row['본전공_전공필수'], 15),
        'req_major_elective': safe_int(row['본전공_전공선택'], 33),
        'req_total': safe_int(row['본전공_계'], 48),
    }


def get_graduation_requirement(
    multi_major: str,
    program_type: str,
    admission_year: int,
    gr_df: pd.DataFrame
) -> Optional[Dict]:
    """다전공 기준 조회"""
    if gr_df.empty:
        return None
    
    result = gr_df[
        (gr_df['전공명'] == multi_major) &
        (gr_df['제도유형'] == program_type) &
        (gr_df['기준학번'] == admission_year)
    ]
    
    if result.empty:
        # 부분 매칭 시도
        keyword = multi_major.replace('전공', '').replace('(평캠)', '').replace('(평택)', '').strip()
        if keyword:
            result = gr_df[
                (gr_df['전공명'].str.contains(keyword, case=False, na=False)) &
                (gr_df['제도유형'] == program_type) &
                (gr_df['기준학번'] == admission_year)
            ]
    
    if result.empty:
        result = gr_df[
            (gr_df['전공명'] == multi_major) &
            (gr_df['제도유형'] == program_type)
        ]
        if not result.empty:
            closest_year = min(result['기준학번'].unique(), key=lambda x: abs(x - admission_year))
            result = result[result['기준학번'] == closest_year]
    
    if result.empty:
        # 기본값 반환
        defaults = {
            "복수전공": (15, 21, 36),
            "부전공": (6, 15, 21),
            "융합전공": (15, 21, 36),
            "융합부전공": (6, 15, 21),
        }
        req, elec, total = defaults.get(program_type, (15, 21, 36))
        return {
            'major_name': multi_major,
            'program_type': program_type,
            'admission_year': admission_year,
            'req_multi_required': req,
            'req_multi_elective': elec,
            'req_total': total,
        }
    
    row = result.iloc[0]
    return {
        'major_name': row['전공명'],
        'program_type': row['제도유형'],
        'admission_year': safe_int(row['기준학번'], admission_year),
        'req_multi_required': safe_int(row['다전공_전공필수'], 15),
        'req_multi_elective': safe_int(row['다전공_전공선택'], 21),
        'req_total': safe_int(row['다전공_계'], 36),
    }


# ============================================================
# 계산 함수
# ============================================================

def get_total_semesters(admission_type: str) -> int:
    """총 학기 수 계산"""
    if admission_type == "신입학":
        return FRESHMAN_TOTAL_SEMESTERS
    else:
        return TRANSFER_TOTAL_SEMESTERS


def calculate_remaining_semesters(admission_type: str, completed_semesters: int) -> int:
    """남은 학기 계산"""
    total = get_total_semesters(admission_type)
    return max(0, total - completed_semesters)


def calculate_max_additional_credits(remaining_semesters: int) -> int:
    """최대 추가 이수 가능 학점"""
    return remaining_semesters * MAX_CREDITS_PER_SEMESTER


def apply_excess_to_elective(
    completed_required: int,
    req_required: int,
    completed_elective: int
) -> Tuple[int, int]:
    """전공필수 초과분 → 전공선택 이월"""
    if completed_required > req_required:
        excess = completed_required - req_required
        return req_required, completed_elective + excess
    return completed_required, completed_elective


def calculate_deficit(completed: int, required: int) -> int:
    """부족 학점 계산"""
    return max(0, required - completed)


def calculate_graduation_credits(
    program_type: str,
    primary_grad_credits: int,
    multi_grad_credits: int
) -> int:
    """제도별 졸업학점 계산"""
    if program_type == "복수전공":
        # 둘 중 큰 값, 최대 130
        return min(max(primary_grad_credits, multi_grad_credits), MAX_DOUBLE_MAJOR_CREDITS)
    else:
        return primary_grad_credits


def determine_graduation_status(
    deficit_total: int,
    max_additional: int,
    deficit_major_required: int,
    deficit_multi_required: int,
    remaining_semesters: int
) -> Tuple[str, bool]:
    """졸업 가능 상태 판단"""
    if deficit_total <= 0:
        return "가능", True
    
    if deficit_total > max_additional:
        return "어려움", False
    
    # 필수 과목 이수 가능 여부
    total_required_deficit = deficit_major_required + deficit_multi_required
    
    # 여유도 계산
    margin = max_additional - deficit_total
    
    if margin >= remaining_semesters * 6:  # 학기당 6학점 이상 여유
        return "가능", True
    elif margin >= 0:
        return "위험", True
    else:
        return "어려움", False


# ============================================================
# 분석 함수
# ============================================================

def analyze_current_status(student: StudentInput, pr_df: pd.DataFrame) -> CreditAnalysis:
    """현재 상태 분석 (본전공 기준, 다전공 미참여 시)"""
    analysis = CreditAnalysis()
    
    # 남은 학기 계산
    analysis.remaining_semesters = calculate_remaining_semesters(
        student.admission_type,
        student.completed_semesters
    )
    analysis.max_additional_credits = calculate_max_additional_credits(analysis.remaining_semesters)
    
    # 기본 졸업학점 (다전공 미참여 시 본전공 기준 = 120학점 가정)
    analysis.req_graduation_credits = DEFAULT_GRADUATION_CREDITS
    
    # 본전공 기준 조회 (복수전공 기준으로 조회하여 기본값 사용)
    pr_req = get_primary_requirement(student.primary_major, "복수전공", student.admission_year, pr_df)
    
    if pr_req:
        analysis.req_major_required = pr_req['req_major_required']
        analysis.req_major_elective = pr_req['req_major_elective']
    else:
        # 기본값
        analysis.req_major_required = 15
        analysis.req_major_elective = 33
    
    # 이수 학점
    analysis.completed_major_required = student.credits_major_required
    analysis.completed_major_elective = student.credits_major_elective
    
    # 전공필수 초과분 이월
    adj_required, adj_elective = apply_excess_to_elective(
        analysis.completed_major_required,
        analysis.req_major_required,
        analysis.completed_major_elective
    )
    
    # 부족 학점
    analysis.deficit_major_required = calculate_deficit(adj_required, analysis.req_major_required)
    analysis.deficit_major_elective = calculate_deficit(adj_elective, analysis.req_major_elective)
    
    # 총 이수 학점
    if student.admission_type == "신입학":
        analysis.completed_total = (
            student.credits_basic_literacy +
            student.credits_basic_science +
            student.credits_core_liberal +
            student.credits_major_required +
            student.credits_major_elective +
            student.credits_free
        )
    else:
        analysis.completed_total = (
            student.transfer_credits +
            student.credits_major_required +
            student.credits_major_elective +
            student.credits_free
        )
    
    # 졸업학점 부족분
    analysis.deficit_graduation = calculate_deficit(
        analysis.completed_total,
        analysis.req_graduation_credits
    )
    
    return analysis


def simulate_program(
    student: StudentInput,
    program_type: str,
    multi_major: str,
    pr_df: pd.DataFrame,
    gr_df: pd.DataFrame
) -> SimulationResult:
    """단일 제도 시뮬레이션"""
    result = SimulationResult(
        program_type=program_type,
        multi_major_name=multi_major
    )
    
    analysis = CreditAnalysis()
    
    # 남은 학기
    analysis.remaining_semesters = calculate_remaining_semesters(
        student.admission_type,
        student.completed_semesters
    )
    analysis.max_additional_credits = calculate_max_additional_credits(analysis.remaining_semesters)
    
    # 본전공 기준 (다전공 참여 시 변화된 기준)
    pr_req = get_primary_requirement(student.primary_major, program_type, student.admission_year, pr_df)
    
    if pr_req:
        analysis.req_major_required = pr_req['req_major_required']
        analysis.req_major_elective = pr_req['req_major_elective']
        analysis.req_major_required_changed = pr_req['req_major_required']
        analysis.req_major_elective_changed = pr_req['req_major_elective']
    else:
        analysis.req_major_required = 15
        analysis.req_major_elective = 33
        analysis.req_major_required_changed = 15
        analysis.req_major_elective_changed = 33
    
    # 다전공 기준
    gr_req = get_graduation_requirement(multi_major, program_type, student.admission_year, gr_df)
    
    if gr_req:
        analysis.req_multi_required = gr_req['req_multi_required']
        analysis.req_multi_elective = gr_req['req_multi_elective']
    else:
        # 기본값
        if program_type == "복수전공":
            analysis.req_multi_required = 15
            analysis.req_multi_elective = 21
        elif program_type == "부전공":
            analysis.req_multi_required = 6
            analysis.req_multi_elective = 15
        elif program_type == "융합전공":
            analysis.req_multi_required = 15
            analysis.req_multi_elective = 21
        else:
            analysis.req_multi_required = 6
            analysis.req_multi_elective = 15
    
    # 졸업학점 계산
    analysis.req_graduation_credits = calculate_graduation_credits(
        program_type,
        DEFAULT_GRADUATION_CREDITS,
        gr_req['req_total'] + DEFAULT_GRADUATION_CREDITS if gr_req else DEFAULT_GRADUATION_CREDITS
    )
    
    # 이수 학점
    analysis.completed_major_required = student.credits_major_required
    analysis.completed_major_elective = student.credits_major_elective
    analysis.completed_multi_required = 0  # 신규 신청자는 0
    analysis.completed_multi_elective = 0
    
    # 전공필수 초과분 이월
    adj_required, adj_elective = apply_excess_to_elective(
        analysis.completed_major_required,
        analysis.req_major_required,
        analysis.completed_major_elective
    )
    
    # 부족 학점
    analysis.deficit_major_required = calculate_deficit(adj_required, analysis.req_major_required)
    analysis.deficit_major_elective = calculate_deficit(adj_elective, analysis.req_major_elective)
    analysis.deficit_multi_required = analysis.req_multi_required
    analysis.deficit_multi_elective = analysis.req_multi_elective
    
    # 총 이수 학점
    if student.admission_type == "신입학":
        analysis.completed_total = (
            student.credits_basic_literacy +
            student.credits_basic_science +
            student.credits_core_liberal +
            student.credits_major_required +
            student.credits_major_elective +
            student.credits_free
        )
    else:
        analysis.completed_total = (
            student.transfer_credits +
            student.credits_major_required +
            student.credits_major_elective +
            student.credits_free
        )
    
    # 졸업학점 부족분
    total_required = (
        analysis.req_major_required +
        analysis.req_major_elective +
        analysis.req_multi_required +
        analysis.req_multi_elective
    )
    
    # 실제 부족 학점 계산
    total_deficit = (
        analysis.deficit_major_required +
        analysis.deficit_major_elective +
        analysis.deficit_multi_required +
        analysis.deficit_multi_elective
    )
    
    analysis.deficit_graduation = calculate_deficit(
        analysis.completed_total,
        analysis.req_graduation_credits
    )
    
    # 졸업 가능 여부 판단
    status, can_grad = determine_graduation_status(
        total_deficit,
        analysis.max_additional_credits,
        analysis.deficit_major_required,
        analysis.deficit_multi_required,
        analysis.remaining_semesters
    )
    
    result.graduation_status = status
    result.can_graduate = can_grad
    result.credit_analysis = analysis
    
    # 학기별 이수 계획 생성
    result.semester_plan = generate_semester_plan(analysis)
    
    return result


def generate_semester_plan(analysis: CreditAnalysis) -> List[Dict]:
    """학기별 이수 계획 생성"""
    plan = []
    
    if analysis.remaining_semesters <= 0:
        return plan
    
    # 총 부족 학점
    total_deficit = (
        analysis.deficit_major_required +
        analysis.deficit_major_elective +
        analysis.deficit_multi_required +
        analysis.deficit_multi_elective +
        max(0, analysis.deficit_graduation - (
            analysis.deficit_major_required +
            analysis.deficit_major_elective +
            analysis.deficit_multi_required +
            analysis.deficit_multi_elective
        ))
    )
    
    # 학기별 균등 배분
    credits_per_semester = min(MAX_CREDITS_PER_SEMESTER, 
                               (total_deficit + analysis.remaining_semesters - 1) // analysis.remaining_semesters)
    
    remaining_major_req = analysis.deficit_major_required
    remaining_major_elec = analysis.deficit_major_elective
    remaining_multi_req = analysis.deficit_multi_required
    remaining_multi_elec = analysis.deficit_multi_elective
    
    for sem in range(1, analysis.remaining_semesters + 1):
        sem_plan = {
            "semester": sem,
            "major_required": 0,
            "major_elective": 0,
            "multi_required": 0,
            "multi_elective": 0,
            "total": 0
        }
        
        remaining_credits = credits_per_semester
        
        # 전공필수 우선 배정
        if remaining_major_req > 0:
            take = min(remaining_major_req, remaining_credits, 6)
            sem_plan["major_required"] = take
            remaining_major_req -= take
            remaining_credits -= take
        
        if remaining_multi_req > 0 and remaining_credits > 0:
            take = min(remaining_multi_req, remaining_credits, 6)
            sem_plan["multi_required"] = take
            remaining_multi_req -= take
            remaining_credits -= take
        
        # 전공선택 배정
        if remaining_major_elec > 0 and remaining_credits > 0:
            take = min(remaining_major_elec, remaining_credits)
            sem_plan["major_elective"] = take
            remaining_major_elec -= take
            remaining_credits -= take
        
        if remaining_multi_elec > 0 and remaining_credits > 0:
            take = min(remaining_multi_elec, remaining_credits)
            sem_plan["multi_elective"] = take
            remaining_multi_elec -= take
            remaining_credits -= take
        
        sem_plan["total"] = (
            sem_plan["major_required"] +
            sem_plan["major_elective"] +
            sem_plan["multi_required"] +
            sem_plan["multi_elective"]
        )
        
        if sem_plan["total"] > 0:
            plan.append(sem_plan)
    
    return plan


def rank_recommendations(results: List[SimulationResult]) -> Tuple[List[SimulationResult], List[SimulationResult]]:
    """추천 순위 정렬"""
    
    def get_score(r: SimulationResult) -> Tuple:
        """정렬 점수 계산 (낮을수록 좋음)"""
        # 1. 졸업 가능 여부 (가능 > 위험 > 어려움)
        grad_score = {"가능": 0, "위험": 1, "어려움": 2}.get(r.graduation_status, 2)
        
        # 2. 총 부족 학점 (±3학점 동일 취급을 위해 3으로 나눔)
        total_deficit = (
            r.credit_analysis.deficit_major_required +
            r.credit_analysis.deficit_major_elective +
            r.credit_analysis.deficit_multi_required +
            r.credit_analysis.deficit_multi_elective
        )
        deficit_score = total_deficit // 3
        
        # 3. 제도 우선순위
        priority_score = PROGRAM_PRIORITY.get(r.program_type, 5)
        
        return (grad_score, deficit_score, priority_score)
    
    # 보조 추천 분리 (연계전공)
    main_results = [r for r in results if r.program_type != "연계전공"]
    supplementary = [r for r in results if r.program_type == "연계전공"]
    
    # 정렬
    main_results.sort(key=get_score)
    supplementary.sort(key=get_score)
    
    # 순위 부여 및 추천 사유 생성
    for idx, r in enumerate(main_results):
        r.recommendation_rank = idx + 1
        r.recommendation_reason = generate_recommendation_reason(r, idx + 1)
    
    for idx, r in enumerate(supplementary):
        r.recommendation_rank = idx + 1
        r.is_supplementary = True
        r.recommendation_reason = generate_recommendation_reason(r, idx + 1, is_supplementary=True)
    
    return main_results, supplementary


def generate_recommendation_reason(result: SimulationResult, rank: int, is_supplementary: bool = False) -> str:
    """추천 사유 생성"""
    reasons = []
    
    analysis = result.credit_analysis
    total_deficit = (
        analysis.deficit_major_required +
        analysis.deficit_major_elective +
        analysis.deficit_multi_required +
        analysis.deficit_multi_elective
    )
    
    if result.graduation_status == "가능":
        reasons.append(f"남은 {analysis.remaining_semesters}학기 내 졸업 가능")
    elif result.graduation_status == "위험":
        reasons.append(f"학기당 집중 이수 시 졸업 가능")
    else:
        reasons.append(f"현재 학점으로는 졸업이 어려움")
    
    if total_deficit <= 36:
        reasons.append(f"총 {total_deficit}학점만 추가 이수 필요")
    else:
        reasons.append(f"총 {total_deficit}학점 추가 이수 필요")
    
    if result.program_type == "복수전공":
        reasons.append("학위 2개 취득 가능")
    elif result.program_type == "부전공":
        reasons.append("비교적 적은 학점으로 이수 가능")
    elif result.program_type == "융합전공":
        reasons.append("융합적 역량 강화")
    elif result.program_type == "융합부전공":
        reasons.append("적은 학점으로 융합 역량 확보")
    
    if is_supplementary:
        reasons.insert(0, "[보조 추천]")
    
    return " / ".join(reasons)


# ============================================================
# 메인 시뮬레이션 함수
# ============================================================

def run_simulation(student: StudentInput) -> AnalysisOutput:
    """통합 시뮬레이션 실행"""
    output = AnalysisOutput()
    output.student_input = student
    
    # 데이터 로드
    pr_df = load_primary_requirements()
    gr_df = load_graduation_requirements()
    
    # 현재 상태 분석
    output.current_analysis = analyze_current_status(student, pr_df)
    
    _, output.current_can_graduate = determine_graduation_status(
        output.current_analysis.deficit_graduation,
        output.current_analysis.max_additional_credits,
        output.current_analysis.deficit_major_required,
        0,
        output.current_analysis.remaining_semesters
    )
    
    if student.student_type == "신규 신청자" and student.desired_multi_major:
        # 모든 제도에 대해 시뮬레이션
        programs = ["복수전공", "부전공", "융합전공", "융합부전공"]
        
        for program in programs:
            result = simulate_program(
                student, program, student.desired_multi_major,
                pr_df, gr_df
            )
            output.simulation_results.append(result)
        
        # 추천 순위 정렬
        output.recommended_programs, output.supplementary_programs = rank_recommendations(
            output.simulation_results
        )
    
    elif student.student_type == "기존 참여자" and student.current_program and student.current_multi_major:
        # 현재 참여 중인 제도만 분석
        result = simulate_program(
            student, student.current_program, student.current_multi_major,
            pr_df, gr_df
        )
        # 기존 참여자의 이수 학점 반영
        result.credit_analysis.completed_multi_required = student.credits_multi_required
        result.credit_analysis.completed_multi_elective = student.credits_multi_elective
        result.credit_analysis.deficit_multi_required = calculate_deficit(
            student.credits_multi_required,
            result.credit_analysis.req_multi_required
        )
        result.credit_analysis.deficit_multi_elective = calculate_deficit(
            student.credits_multi_elective,
            result.credit_analysis.req_multi_elective
        )
        
        output.simulation_results.append(result)
    
    return output


# ============================================================
# Streamlit UI 함수
# ============================================================

def render_simulation_page():
    """다전공 추천 시뮬레이션 페이지"""
    
    st.markdown("""
    <h1 style="text-align: center; color: #667eea; margin-bottom: 10px;">
        🎯 다전공 추천 시뮬레이션
    </h1>
    <p style="text-align: center; color: #666; margin-bottom: 30px;">
        학점 현황을 입력하고 나에게 맞는 다전공 제도를 찾아보세요!
    </p>
    """, unsafe_allow_html=True)
    
    # 진행 단계 표시
    if 'sim_step' not in st.session_state:
        st.session_state.sim_step = 1
    
    # 탭 대신 단계별 진행
    col1, col2, col3, col4 = st.columns(4)
    
    steps = [
        ("1️⃣", "유형 선택"),
        ("2️⃣", "기본 정보"),
        ("3️⃣", "학점 입력"),
        ("4️⃣", "결과 확인")
    ]
    
    for idx, (col, (emoji, label)) in enumerate(zip([col1, col2, col3, col4], steps)):
        with col:
            if idx + 1 == st.session_state.sim_step:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 10px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem;">{emoji}</div>
                    <div style="font-size: 0.85rem; font-weight: bold;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            elif idx + 1 < st.session_state.sim_step:
                st.markdown(f"""
                <div style="background: #28a745; color: white; padding: 10px; 
                            border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem;">✅</div>
                    <div style="font-size: 0.85rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #e9ecef; color: #666; padding: 10px; 
                            border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem;">{emoji}</div>
                    <div style="font-size: 0.85rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 단계별 컨텐츠
    if st.session_state.sim_step == 1:
        render_step1_student_type()
    elif st.session_state.sim_step == 2:
        render_step2_basic_info()
    elif st.session_state.sim_step == 3:
        render_step3_credits()
    elif st.session_state.sim_step == 4:
        render_step4_results()


def render_step1_student_type():
    """STEP 1: 학생 유형 선택"""
    
    st.markdown("""
    <div style="background: white; border-radius: 15px; padding: 30px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #333; margin-bottom: 20px;">📋 어떤 상황인가요?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🆕 다전공을 새로 신청하려고 해요", use_container_width=True, type="primary"):
            st.session_state.sim_student_type = "신규 신청자"
            st.session_state.sim_step = 2
            st.rerun()
        
        st.markdown("""
        <p style="color: #666; font-size: 0.9rem; text-align: center; margin-top: 10px;">
            아직 다전공에 참여하지 않았고,<br>어떤 제도가 좋을지 알고 싶어요
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📚 이미 다전공을 하고 있어요", use_container_width=True):
            st.session_state.sim_student_type = "기존 참여자"
            st.session_state.sim_step = 2
            st.rerun()
        
        st.markdown("""
        <p style="color: #666; font-size: 0.9rem; text-align: center; margin-top: 10px;">
            이미 다전공에 참여 중이고,<br>남은 학점을 확인하고 싶어요
        </p>
        """, unsafe_allow_html=True)


def render_step2_basic_info():
    """STEP 2: 기본 정보 입력"""
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                border-radius: 15px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #667eea; margin: 0;">
            {'🆕 신규 신청자' if st.session_state.sim_student_type == '신규 신청자' else '📚 기존 참여자'} - 기본 정보
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    majors = load_majors_list()
    if not majors:
        majors = ["경영학전공", "컴퓨터공학전공", "영미언어문화전공"]  # 기본값
    
    col1, col2 = st.columns(2)
    
    with col1:
        admission_year = st.selectbox(
            "📅 입학연도",
            options=list(range(2025, 2019, -1)),
            help="학번 기준 연도를 선택하세요"
        )
        
        primary_major = st.selectbox(
            "🎓 본전공",
            options=majors,
            help="현재 소속된 전공을 선택하세요"
        )
    
    with col2:
        admission_type = st.selectbox(
            "📝 입학구분",
            options=["신입학", "3학년 편입학(동일계)", "3학년 편입학(비동일계)"]
        )
        
        max_sem = 8 if admission_type == "신입학" else 4
        completed_semesters = st.selectbox(
            "📆 현재까지 이수한 학기 수",
            options=list(range(1, max_sem + 1)),
            help="휴학 학기 제외"
        )
    
    # 편입생 인정학점
    transfer_credits = 0
    if admission_type != "신입학":
        st.markdown("---")
        transfer_credits = st.number_input(
            "🔢 편입학 인정학점",
            min_value=0,
            max_value=70,
            value=65,
            help="편입 시 인정받은 학점을 입력하세요"
        )
    
    # 다전공 정보 (신규 신청자)
    desired_multi_major = None
    if st.session_state.sim_student_type == "신규 신청자":
        st.markdown("---")
        st.markdown("### 🎯 희망 다전공")
        
        # 다전공 목록 (복수전공 기준으로 가져옴)
        multi_majors = load_multi_majors_by_program("복수전공")
        if not multi_majors:
            multi_majors = majors
        
        desired_multi_major = st.selectbox(
            "다전공으로 이수하고 싶은 전공",
            options=multi_majors,
            help="시뮬레이션할 다전공을 선택하세요"
        )
    
    # 기존 참여자 정보
    current_program = None
    current_multi_major = None
    if st.session_state.sim_student_type == "기존 참여자":
        st.markdown("---")
        st.markdown("### 📚 현재 참여 중인 다전공")
        
        col1, col2 = st.columns(2)
        with col1:
            current_program = st.selectbox(
                "참여 중인 제도",
                options=["복수전공", "부전공", "융합전공", "융합부전공", "연계전공"]
            )
        
        with col2:
            multi_majors = load_multi_majors_by_program(current_program)
            if not multi_majors:
                multi_majors = majors
            current_multi_major = st.selectbox(
                "참여 중인 다전공명",
                options=multi_majors
            )
    
    # 세션에 저장
    st.session_state.sim_admission_year = admission_year
    st.session_state.sim_primary_major = primary_major
    st.session_state.sim_admission_type = admission_type
    st.session_state.sim_completed_semesters = completed_semesters
    st.session_state.sim_transfer_credits = transfer_credits
    st.session_state.sim_desired_multi_major = desired_multi_major
    st.session_state.sim_current_program = current_program
    st.session_state.sim_current_multi_major = current_multi_major
    
    # 네비게이션 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.sim_step = 1
            st.rerun()
    
    with col3:
        if st.button("다음 ➡️", use_container_width=True, type="primary"):
            st.session_state.sim_step = 3
            st.rerun()


def render_step3_credits():
    """STEP 3: 학점 입력"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                border-radius: 15px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #667eea; margin: 0;">📊 현재 이수 학점 입력</h3>
        <p style="color: #666; margin: 10px 0 0 0; font-size: 0.9rem;">
            정확한 분석을 위해 현재까지 이수한 학점을 입력해주세요
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 교양 학점 (신입학만)
    credits_basic_literacy = 0
    credits_basic_science = 0
    credits_core_liberal = 0
    
    if st.session_state.sim_admission_type == "신입학":
        st.markdown("### 📚 교양 학점")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            credits_basic_literacy = st.number_input(
                "기초교양(기초문해)",
                min_value=0, max_value=30, value=0,
                help="기초문해 영역 이수 학점"
            )
        
        with col2:
            credits_basic_science = st.number_input(
                "기초교양(기초과학)",
                min_value=0, max_value=30, value=0,
                help="기초과학 영역 이수 학점 (해당 시)"
            )
        
        with col3:
            credits_core_liberal = st.number_input(
                "핵심교양",
                min_value=0, max_value=30, value=0,
                help="핵심교양 영역 이수 학점"
            )
    
    # 본전공 학점
    st.markdown("### 🎓 본전공 학점")
    col1, col2 = st.columns(2)
    
    with col1:
        credits_major_required = st.number_input(
            "전공필수 이수 학점",
            min_value=0, max_value=60, value=0,
            help="본전공 전공필수 이수 학점"
        )
    
    with col2:
        credits_major_elective = st.number_input(
            "전공선택 이수 학점",
            min_value=0, max_value=60, value=0,
            help="본전공 전공선택 이수 학점"
        )
    
    # 다전공 학점 (기존 참여자만)
    credits_multi_required = 0
    credits_multi_elective = 0
    
    if st.session_state.sim_student_type == "기존 참여자":
        st.markdown(f"### 📘 다전공 학점 ({st.session_state.sim_current_program})")
        col1, col2 = st.columns(2)
        
        with col1:
            credits_multi_required = st.number_input(
                "다전공 전공필수 이수 학점",
                min_value=0, max_value=60, value=0,
                help="다전공 전공필수 이수 학점"
            )
        
        with col2:
            credits_multi_elective = st.number_input(
                "다전공 전공선택 이수 학점",
                min_value=0, max_value=60, value=0,
                help="다전공 전공선택 이수 학점"
            )
    
    # 잔여 학점
    st.markdown("### 📋 기타 학점")
    credits_free = st.number_input(
        "잔여(자유) 이수 학점",
        min_value=0, max_value=60, value=0,
        help="소양교양, 자유선택 등 기타 이수 학점"
    )
    
    # 총 이수 학점 미리보기
    if st.session_state.sim_admission_type == "신입학":
        total = (credits_basic_literacy + credits_basic_science + credits_core_liberal +
                credits_major_required + credits_major_elective + 
                credits_multi_required + credits_multi_elective + credits_free)
    else:
        total = (st.session_state.sim_transfer_credits +
                credits_major_required + credits_major_elective +
                credits_multi_required + credits_multi_elective + credits_free)
    
    st.markdown(f"""
    <div style="background: #e3f2fd; border-radius: 10px; padding: 15px; margin-top: 20px;">
        <h4 style="color: #1565c0; margin: 0;">📊 총 이수 학점: <span style="font-size: 1.5rem;">{total}</span>학점</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # 세션에 저장
    st.session_state.sim_credits_basic_literacy = credits_basic_literacy
    st.session_state.sim_credits_basic_science = credits_basic_science
    st.session_state.sim_credits_core_liberal = credits_core_liberal
    st.session_state.sim_credits_major_required = credits_major_required
    st.session_state.sim_credits_major_elective = credits_major_elective
    st.session_state.sim_credits_multi_required = credits_multi_required
    st.session_state.sim_credits_multi_elective = credits_multi_elective
    st.session_state.sim_credits_free = credits_free
    
    # 네비게이션 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.sim_step = 2
            st.rerun()
    
    with col3:
        if st.button("🔍 분석하기", use_container_width=True, type="primary"):
            st.session_state.sim_step = 4
            st.rerun()


def render_step4_results():
    """STEP 4: 결과 확인"""
    
    # StudentInput 객체 생성
    student = StudentInput(
        student_type=st.session_state.sim_student_type,
        admission_year=st.session_state.sim_admission_year,
        primary_major=st.session_state.sim_primary_major,
        admission_type=st.session_state.sim_admission_type,
        completed_semesters=st.session_state.sim_completed_semesters,
        transfer_credits=st.session_state.get('sim_transfer_credits', 0),
        credits_basic_literacy=st.session_state.get('sim_credits_basic_literacy', 0),
        credits_basic_science=st.session_state.get('sim_credits_basic_science', 0),
        credits_core_liberal=st.session_state.get('sim_credits_core_liberal', 0),
        credits_major_required=st.session_state.get('sim_credits_major_required', 0),
        credits_major_elective=st.session_state.get('sim_credits_major_elective', 0),
        credits_free=st.session_state.get('sim_credits_free', 0),
        desired_multi_major=st.session_state.get('sim_desired_multi_major'),
        current_program=st.session_state.get('sim_current_program'),
        current_multi_major=st.session_state.get('sim_current_multi_major'),
        credits_multi_required=st.session_state.get('sim_credits_multi_required', 0),
        credits_multi_elective=st.session_state.get('sim_credits_multi_elective', 0),
    )
    
    # 시뮬레이션 실행
    output = run_simulation(student)
    
    # 결과 헤더
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; padding: 25px; margin-bottom: 20px; color: white;">
        <h2 style="margin: 0; color: white;">📊 분석 결과</h2>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">
            입력하신 정보를 바탕으로 분석한 결과입니다
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 기본 정보 요약
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎓 본전공", student.primary_major)
    with col2:
        st.metric("📅 입학연도", f"{student.admission_year}학번")
    with col3:
        remaining = output.current_analysis.remaining_semesters
        st.metric("⏳ 남은 학기", f"{remaining}학기")
    
    st.markdown("---")
    
    # 현재 상태 분석
    st.markdown("### 📈 현재 상태 (본전공 기준)")
    
    analysis = output.current_analysis
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 학점 현황 카드
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 20px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h4 style="color: #333; margin-bottom: 15px;">📚 학점 현황</h4>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공필수</td>
                    <td style="text-align: right; font-weight: bold;">{analysis.completed_major_required} / {analysis.req_major_required} 학점</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공선택</td>
                    <td style="text-align: right; font-weight: bold;">{analysis.completed_major_elective} / {analysis.req_major_elective} 학점</td>
                </tr>
                <tr style="border-top: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #333; font-weight: bold;">총 이수</td>
                    <td style="text-align: right; font-weight: bold; color: #667eea; font-size: 1.1rem;">
                        {analysis.completed_total} / {analysis.req_graduation_credits} 학점
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 부족 학점 카드
        total_deficit = analysis.deficit_major_required + analysis.deficit_major_elective
        grad_color = "#28a745" if output.current_can_graduate else "#dc3545"
        grad_text = "졸업 가능" if output.current_can_graduate else "학점 부족"
        
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 20px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h4 style="color: #333; margin-bottom: 15px;">⚠️ 부족 학점</h4>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공필수 부족</td>
                    <td style="text-align: right; font-weight: bold; color: {'#dc3545' if analysis.deficit_major_required > 0 else '#28a745'};">
                        {analysis.deficit_major_required} 학점
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공선택 부족</td>
                    <td style="text-align: right; font-weight: bold; color: {'#dc3545' if analysis.deficit_major_elective > 0 else '#28a745'};">
                        {analysis.deficit_major_elective} 학점
                    </td>
                </tr>
                <tr style="border-top: 1px solid #eee;">
                    <td style="padding: 12px 0; color: #333; font-weight: bold;">상태</td>
                    <td style="text-align: right; font-weight: bold; color: {grad_color}; font-size: 1.1rem;">
                        {grad_text}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # 신규 신청자: 제도별 시뮬레이션 결과
    if student.student_type == "신규 신청자" and output.recommended_programs:
        st.markdown("---")
        st.markdown(f"### 🎯 다전공 제도별 시뮬레이션 ({student.desired_multi_major})")
        
        # 추천 순위
        for idx, result in enumerate(output.recommended_programs):
            render_simulation_result_card(result, idx == 0)
        
        # 보조 추천
        if output.supplementary_programs:
            with st.expander("📎 보조 추천 (연계전공)"):
                for result in output.supplementary_programs:
                    render_simulation_result_card(result, False)
    
    # 기존 참여자: 현재 참여 중인 제도 분석
    elif student.student_type == "기존 참여자" and output.simulation_results:
        st.markdown("---")
        st.markdown(f"### 📚 현재 참여 중인 다전공 분석 ({student.current_program})")
        
        result = output.simulation_results[0]
        render_current_participant_analysis(result, student)
    
    # 처음으로 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ 이전", use_container_width=True):
            st.session_state.sim_step = 3
            st.rerun()
    
    with col3:
        if st.button("🔄 처음부터", use_container_width=True):
            # 세션 초기화
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('sim_')]
            for k in keys_to_delete:
                del st.session_state[k]
            st.session_state.sim_step = 1
            st.rerun()


def render_simulation_result_card(result: SimulationResult, is_top: bool):
    """시뮬레이션 결과 카드 렌더링"""
    
    analysis = result.credit_analysis
    
    # 색상 설정
    if result.can_graduate:
        if result.graduation_status == "가능":
            status_color = "#28a745"
            status_bg = "#d4edda"
            status_icon = "✅"
        else:
            status_color = "#ffc107"
            status_bg = "#fff3cd"
            status_icon = "⚠️"
    else:
        status_color = "#dc3545"
        status_bg = "#f8d7da"
        status_icon = "❌"
    
    border_style = "3px solid #667eea" if is_top else "1px solid #e9ecef"
    
    st.markdown(f"""
<div style="background: white; border-radius: 12px; padding: 20px; 
margin-bottom: 15px; border-left: {border_style};
box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
<div>
<span style="font-size: 1.3rem; font-weight: bold; color: #333;">
{result.program_type}
</span>
{f'<span style="background: #667eea; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; margin-left: 10px;">👑 추천 1위</span>' if is_top else ''}
</div>
<span style="background: {status_bg}; color: {status_color}; padding: 5px 15px; 
border-radius: 20px; font-weight: bold;">
{status_icon} {result.graduation_status}
</span>
</div>       
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">
<div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px;">
<div style="font-size: 0.85rem; color: #666;">본전공 필수</div>
<div style="font-size: 1.2rem; font-weight: bold; color: {'#dc3545' if analysis.deficit_major_required > 0 else '#28a745'};">
-{analysis.deficit_major_required}
</div>
</div>
<div style="text-align: center; padding: 10px; background: #f8f9fa; border-radius: 8px;">
<div style="font-size: 0.85rem; color: #666;">본전공 선택</div>
<div style="font-size: 1.2rem; font-weight: bold; color: {'#dc3545' if analysis.deficit_major_elective > 0 else '#28a745'};">
-{analysis.deficit_major_elective}
</div>
</div>
<div style="text-align: center; padding: 10px; background: #e3f2fd; border-radius: 8px;">
<div style="font-size: 0.85rem; color: #666;">다전공 필수</div>
<div style="font-size: 1.2rem; font-weight: bold; color: #1565c0;">
{analysis.req_multi_required}
</div>
</div>
<div style="text-align: center; padding: 10px; background: #e3f2fd; border-radius: 8px;">
<div style="font-size: 0.85rem; color: #666;">다전공 선택</div>
<div style="font-size: 1.2rem; font-weight: bold; color: #1565c0;">
{analysis.req_multi_elective}
</div>
</div>
</div>
<div style="background: #f8f9fa; border-radius: 8px; padding: 12px;">
<span style="color: #666;">💡 </span>
<span style="color: #333;">{result.recommendation_reason}</span>
</div>
</div>
""", unsafe_allow_html=True)
    
    # 학기별 이수 계획 (펼치기)
    if result.semester_plan:
        with st.expander(f"📅 {result.program_type} 학기별 이수 계획"):
            render_semester_plan_table(result.semester_plan)


def render_current_participant_analysis(result: SimulationResult, student: StudentInput):
    """기존 참여자 분석 결과 렌더링"""
    
    analysis = result.credit_analysis
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 20px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h4 style="color: #667eea; margin-bottom: 15px;">🎓 본전공 ({student.primary_major})</h4>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공필수</td>
                    <td style="text-align: right;">{student.credits_major_required} / {analysis.req_major_required} 학점</td>
                    <td style="text-align: right; color: {'#dc3545' if analysis.deficit_major_required > 0 else '#28a745'}; font-weight: bold;">
                        {'부족 ' + str(analysis.deficit_major_required) if analysis.deficit_major_required > 0 else '✓'}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공선택</td>
                    <td style="text-align: right;">{student.credits_major_elective} / {analysis.req_major_elective} 학점</td>
                    <td style="text-align: right; color: {'#dc3545' if analysis.deficit_major_elective > 0 else '#28a745'}; font-weight: bold;">
                        {'부족 ' + str(analysis.deficit_major_elective) if analysis.deficit_major_elective > 0 else '✓'}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; border-radius: 12px; padding: 20px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
            <h4 style="color: #764ba2; margin-bottom: 15px;">📘 다전공 ({student.current_multi_major})</h4>
            <table style="width: 100%;">
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공필수</td>
                    <td style="text-align: right;">{student.credits_multi_required} / {analysis.req_multi_required} 학점</td>
                    <td style="text-align: right; color: {'#dc3545' if analysis.deficit_multi_required > 0 else '#28a745'}; font-weight: bold;">
                        {'부족 ' + str(analysis.deficit_multi_required) if analysis.deficit_multi_required > 0 else '✓'}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666;">전공선택</td>
                    <td style="text-align: right;">{student.credits_multi_elective} / {analysis.req_multi_elective} 학점</td>
                    <td style="text-align: right; color: {'#dc3545' if analysis.deficit_multi_elective > 0 else '#28a745'}; font-weight: bold;">
                        {'부족 ' + str(analysis.deficit_multi_elective) if analysis.deficit_multi_elective > 0 else '✓'}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # 졸업 가능 여부
    total_deficit = (analysis.deficit_major_required + analysis.deficit_major_elective +
                    analysis.deficit_multi_required + analysis.deficit_multi_elective)
    
    status_color = "#28a745" if result.can_graduate else "#dc3545"
    status_text = "졸업 가능" if result.can_graduate else "학점 부족"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {status_color}15 0%, {status_color}05 100%); 
                border-left: 4px solid {status_color}; border-radius: 12px; 
                padding: 20px; margin-top: 20px;">
        <h4 style="color: {status_color}; margin: 0 0 10px 0;">
            {'✅' if result.can_graduate else '⚠️'} {status_text}
        </h4>
        <p style="color: #666; margin: 0;">
            남은 학기: <strong>{analysis.remaining_semesters}학기</strong> / 
            추가 이수 필요: <strong>{total_deficit}학점</strong> /
            학기당 평균: <strong>{total_deficit // max(1, analysis.remaining_semesters)}학점</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # 학기별 이수 계획
    if result.semester_plan:
        st.markdown("### 📅 학기별 이수 계획")
        render_semester_plan_table(result.semester_plan)


def render_semester_plan_table(plan: List[Dict]):
    """학기별 이수 계획 테이블"""
    
    if not plan:
        st.info("이수할 학점이 없습니다.")
        return
    
    # DataFrame으로 변환
    df = pd.DataFrame(plan)
    df.columns = ['학기', '본전공 필수', '본전공 선택', '다전공 필수', '다전공 선택', '합계']
    df['학기'] = df['학기'].apply(lambda x: f"{x}학기")
    
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "학기": st.column_config.TextColumn("학기", width="small"),
            "본전공 필수": st.column_config.NumberColumn("본전공 필수", format="%d학점"),
            "본전공 선택": st.column_config.NumberColumn("본전공 선택", format="%d학점"),
            "다전공 필수": st.column_config.NumberColumn("다전공 필수", format="%d학점"),
            "다전공 선택": st.column_config.NumberColumn("다전공 선택", format="%d학점"),
            "합계": st.column_config.NumberColumn("합계", format="%d학점"),
        }
    )


# ============================================================
# 메인 함수 (테스트용)
# ============================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="다전공 추천 시뮬레이션",
        page_icon="🎯",
        layout="wide"
    )
    render_simulation_page()
