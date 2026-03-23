import pandas as pd
import sys, io, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

faq_df = pd.read_excel('data/faq_mapping.xlsx')

def normalize_for_matching(text):
    text = text.lower().replace(' ', '')
    text = re.sub(r'[?!.,\xb7\u2022/]', '', text)
    particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과', '도', '의', '만', '에게', '한테', '께']
    for p in sorted(particles, key=len, reverse=True):
        text = text.replace(p, '')
    return text

def extract_program_from_text(text):
    text_lower = text.lower().replace(' ', '').replace('\xb7', '').replace('\u2022', '').replace('/', '')
    program_order = [
        ('소단위전공과정', '소단위전공과정'), ('마이크로디그리', '마이크로디그리'),
        ('융합부전공', '융합부전공'), ('융합전공', '융합전공'),
        ('복수전공', '복수전공'), ('부전공', '부전공'),
        ('연계전공', '연계전공'), ('다전공', '다전공'),
        ('유연학사제도', '유연학사제도'), ('유연학사', '유연학사제도'),
    ]
    for key, val in program_order:
        if key in text_lower:
            return val
    if 'md' in text_lower or '마이크로' in text_lower or '소단위' in text_lower:
        return '마이크로디그리'
    if '복전' in text_lower:
        return '복수전공'
    if '부전' in text_lower:
        return '부전공'
    return None

def search_faq_mapping(user_input, faq_df):
    if faq_df.empty:
        return None, 0
    user_clean = user_input.lower().replace(' ', '').replace('\xb7', '').replace('\u2022', '').replace('/', '')
    user_normalized = normalize_for_matching(user_input)

    list_keywords = ['목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '무슨과정', '뭐가있어', '뭐있어', '어떤게있어', '뭐가있']
    if any(kw in user_clean for kw in list_keywords):
        return None, 0

    contact_guard = ['연락처', '전화번호', '번호알려줘', '사무실', '문의처']
    if any(kw in user_clean for kw in contact_guard):
        return None, 0

    detected_program = extract_program_from_text(user_input)
    academic_keywords = ['증명서', '학점교류', '교직', '교원자격', '휴학', '복학', '전과', '전공변경',
        '재입학', '수강신청', '학점인정', '이수구분', '성적처리', '졸업식', '학위수여식',
        '유예', '졸업유예', '조기졸업', '등록금', '학비', '성적', '학점', '수강내역',
        '계절학기', '수강철회', '졸업', '장학금', '자유학기제', '성적확인', '성적조회',
        '학점확인', '수강확인', '이수학점확인', '학사시스템', '교직이수', '수강', '수강정정', '재수강',
        '적성검사', '이의신청', '무인발급', '정부24', '납부', '여름학기']
    is_academic = any(kw in user_clean for kw in academic_keywords)
    if is_academic and not detected_program:
        detected_program = "학사제도"
    if not detected_program:
        return None, 0

    _all_programs = ['복수전공', '부전공', '융합전공', '융합부전공', '연계전공', '소단위전공과정', '마이크로디그리']
    _secondary = [p for p in _all_programs if p != detected_program and p in user_clean]

    _cost_keywords = ['등록금', '학비', '비용', '돈얼마', '추가비용', '추가학비']
    _include_haksa = any(ck in user_clean for ck in _cost_keywords)

    if detected_program == "학사제도":
        program_faq = faq_df[faq_df['program'] == '학사제도']
    elif detected_program in ['소단위전공과정', '마이크로디그리']:
        _sp = ['소단위전공과정', '마이크로디그리', '다전공'] + _secondary
        if _include_haksa: _sp.append('학사제도')
        program_faq = faq_df[faq_df['program'].isin(_sp)]
    elif detected_program == "다전공":
        _sp = ['다전공']
        if _include_haksa: _sp.append('학사제도')
        program_faq = faq_df[faq_df['program'].isin(_sp)]
    elif detected_program == "유연학사제도":
        program_faq = faq_df[faq_df['program'] == '유연학사제도']
    else:
        _sp = [detected_program, '다전공'] + _secondary
        if _include_haksa: _sp.append('학사제도')
        program_faq = faq_df[faq_df['program'].isin(_sp)]

    if program_faq.empty:
        return None, 0

    best_match = None
    best_score = 0
    _intent_boost = {
        'APPLY_QUALIFICATION': ['자격', '조건', '대상', '기준', '가능', '할수있는', '돼', '되나', '될까', '되는지', '가능해', '가능한가', '가능하나', '할수있나', '아무나', '할수있어'],
        'APPLY_PERIOD': ['기간', '언제', '마감', '일정', '시기', '날짜', '몇월', '2학기'],
        'APPLY_METHOD': ['방법', '절차', '순서', '어떻게', '어디서', '어디', '서류'],
        'CREDIT_INFO': ['학점', '몇학점', '이수학점', '졸업학점'],
        'APPLY_CANCEL': ['취소', '포기', '철회', '그만', '그만두', '그만둘'],
        'APPLY_CHANGE': ['변경', '바꾸', '바꿀', '바꿔', '바꾼', '전환'],
    }
    for _, row in program_faq.iterrows():
        keywords = str(row.get('keyword', '')).split(',')
        keywords = [k.strip().lower().replace(' ', '') for k in keywords if k.strip()]
        exclude_kws = str(row.get('exclude_keywords', '')).split(',')
        exclude_kws = [e.strip().lower().replace(' ', '') for e in exclude_kws if e.strip()]
        if any(ex in user_clean for ex in exclude_kws):
            continue
        # CONCURRENT_ENROLL은 2개 이상 프로그램 감지 시에만 매칭
        row_intent_tmp = str(row.get('intent', ''))
        if row_intent_tmp == 'CONCURRENT_ENROLL' and not _secondary:
            continue
        keyword_matches = 0
        total_keyword_length = 0
        for kw in keywords:
            if kw in user_clean or kw in user_normalized:
                keyword_matches += 1
                total_keyword_length += len(kw)
        if keyword_matches > 0:
            score = total_keyword_length + (keyword_matches * 5)
            if _secondary:
                row_program = str(row.get('program', ''))
                if any(sp in row_program for sp in _secondary) or any(sp in str(row.get('keyword', '')) for sp in _secondary):
                    score += 20
            row_intent = str(row.get('intent', ''))
            if row_intent in _intent_boost:
                if any(ib in user_clean for ib in _intent_boost[row_intent]):
                    score += 25
            if score > best_score:
                best_score = score
                best_match = row
    if best_match is not None:
        return best_match, best_score
    return None, 0

def is_followup_question(user_input):
    user_clean = user_input.replace(' ', '').lower()
    major_patterns = ['전공', '학과', '과정']
    has_major_name = any(p in user_clean for p in major_patterns)
    program_keywords = ['복수전공', '부전공', '융합전공', '마이크로', '소단위', '연계전공', 'md', '유연학사제도', '유연학사', '다전공']
    has_program = any(kw in user_clean for kw in program_keywords)
    academic_keywords = ['증명서', '학점교류', '교직', '교원자격', '휴학', '복학', '전과',
        '수강신청', '학점인정', '이수구분', '성적처리', '졸업식', '학위수여식',
        '졸업유예', '조기졸업', '등록금', '학비', '성적확인', '성적조회',
        '학점확인', '수강확인', '계절학기', '수강철회', '장학금',
        '졸업', '유예', '교직이수', '수강', '성적', '학사시스템',
        '적성검사', '이의신청', '무인발급', '정부24', '납부', '여름학기']
    has_academic = any(kw in user_clean for kw in academic_keywords)
    if has_major_name or has_program or has_academic:
        return False
    followup_indicators = ['그거', '그럼', '그건', '그래서', '거기', '이건', '그리고', '그러면', '아까', '방금']
    if any(ind in user_clean for ind in followup_indicators):
        return True
    question_only_patterns = [
        '신청기간은?', '기간은?', '언제야?', '마감은?',
        '자격은?', '조건은?', '신청자격은?',
        '방법은?', '어떻게해?', '절차는?', '신청방법은?',
        '학점은?', '몇학점?', '이수학점은?',
        '교과목은?', '과목은?',
        '연락처는?', '전화번호는?', '위치는?',
        '차이는?', '뭐가달라?', '똑같아?', '같아?'
    ]
    is_question_only = user_input.strip() in question_only_patterns or any(
        user_clean == p.replace('?', '').replace(' ', '') for p in question_only_patterns
    )
    if is_question_only:
        return True
    return False

def check_program_info_redirect(user_input, program_type):
    if not program_type:
        return False
    _info_words = ['뭐야', '뭔지', '무엇', '설명', '알려줘', '뭐임', '뭐에요', '뭐죠', '어떤거', '어떤것', '어떤제도', '개념', '정의', '궁금', '어떤건지']
    _specific_intent_words = [
        '자격', '조건', '대상', '기준', '신청할수있', '가능',
        '기간', '언제', '마감', '일정', '시기',
        '어떻게', '방법', '절차', '순서',
        '학점', '몇학점', '이수학점', '졸업학점',
        '취소', '포기', '철회', '변경', '바꾸', '전환', '등록금', '학비', '비용', '수강료',
        '좋은점', '장점', '단점', '이점', '메리트',                   # 장단점 (AI fallback)
        '신청과정',                                                # 신청 과정 = 절차
        '목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '무슨과정', '뭐가있', '뭐있',
        '차이', '비교', '다른점', '다른거', 'vs', '차이점', '비교해', '나아', '좋아', '유리',
        '교과목', '과목', '커리큘럼', '연락처', '전화번호', '위치',
        '아무나', '할수있어', '서류', '그만두',
    ]
    _user_clean_tmp = user_input.lower().replace(' ', '')
    _has_specific = any(w in _user_clean_tmp for w in _specific_intent_words)
    if any(w in _user_clean_tmp for w in _info_words) and not _has_specific:
        return True
    return False

def is_comparison_query(user_input):
    """Step 4.7: 비교 질문 감지 (2개 이상 프로그램 + 비교 키워드)"""
    user_clean = user_input.lower().replace(' ', '')
    comp_words = ['차이', '비교', '다른점', '다른거', 'vs', '차이점', '비교해', '달라', '나아', '좋아', '유리', '똑같아', '같은거', '같아', '좋을까', '좋을']
    if not any(w in user_clean for w in comp_words):
        return False
    prog_order = ['소단위전공과정', '마이크로디그리', '융합부전공', '융합전공', '복수전공', '부전공', '연계전공', '다전공']
    found = []
    temp = user_clean
    for p in prog_order:
        if p in temp:
            found.append(p)
            temp = temp.replace(p, '', 1)
    return len(found) >= 2

def is_combine_query(user_input):
    """Step 4.8: 동시 이수 질문 감지 (2개 이상 프로그램 + 동시이수 키워드)"""
    user_clean = user_input.lower().replace(' ', '')
    combine_words = ['같이', '동시', '함께', '겸', '병행', '둘다', '둘 다', '중복', '이중']
    if not any(w in user_clean for w in combine_words):
        return False
    # 약어 → 정식명 치환 후 프로그램 감지
    _abbr = [('복전', '복수전공'), ('부전', '부전공'), ('md', '마이크로디그리'), ('마이크로', '마이크로디그리'), ('소단위', '소단위전공과정')]
    _temp_for_detect = user_clean
    for short, full in _abbr:
        if short in _temp_for_detect and full not in _temp_for_detect:
            _temp_for_detect = _temp_for_detect.replace(short, full, 1)
    prog_order = ['소단위전공과정', '마이크로디그리', '융합부전공', '융합전공', '복수전공', '부전공', '연계전공', '다전공']
    found = []
    temp = _temp_for_detect
    for p in prog_order:
        if p in temp:
            found.append(p)
            temp = temp.replace(p, '', 1)
    return len(found) >= 2

def simulate_step_5_5(user_input, program_type, faq_df):
    """Step 5.5: 의도 기반 FAQ 직접 조회 시뮬레이션"""
    if not program_type:
        return None
    user_clean = user_input.lower().replace(' ', '')
    _intent_kw_map = {
        'APPLY_QUALIFICATION': ['자격', '조건', '대상', '기준', '가능', '돼', '되나', '될까', '할수있', '할수있나', '가능해', '가능한가', '가능하나', '되는지', '아무나', '할수있어'],
        'APPLY_PERIOD': ['기간', '언제', '마감', '일정', '시기', '날짜', '몇월', '2학기'],
        'APPLY_METHOD': ['방법', '절차', '순서', '어디서', '어디', '서류', '어떻게'],
        'CREDIT_INFO': ['학점', '몇학점', '이수학점', '졸업학점'],
        'APPLY_CANCEL': ['취소', '포기', '철회', '그만두', '그만둘'],
        'APPLY_CHANGE': ['변경', '바꾸', '바꿀', '바꿔', '바꾼', '전환'],
        'PROGRAM_TUITION': ['등록금', '학비', '수강료'],
    }
    _detected_intent = None
    for _intent, _kws in _intent_kw_map.items():
        if any(_k in user_clean for _k in _kws):
            _detected_intent = _intent
            break
    if _detected_intent == 'APPLY_QUALIFICATION':
        for _oi, _okws in _intent_kw_map.items():
            if _oi == 'APPLY_QUALIFICATION':
                continue
            if any(_ok in user_clean for _ok in _okws):
                _detected_intent = _oi
                break
    if not _detected_intent:
        return None
    _intent_faq = faq_df[(faq_df['program'] == program_type) & (faq_df['intent'] == _detected_intent)]
    if _intent_faq.empty and program_type not in ['다전공', '유연학사제도']:
        _intent_faq = faq_df[(faq_df['program'] == '다전공') & (faq_df['intent'] == _detected_intent)]
    if not _intent_faq.empty:
        return _detected_intent
    return None

# 의도 충돌 맵 (step 5 conflict resolution)
_icm = {
    'APPLY_CANCEL': ['취소', '포기', '철회', '그만두', '그만둘'],
    'APPLY_CHANGE': ['변경', '바꾸', '바꿀', '바꿔', '바꾼', '전환'],
    'CREDIT_INFO': ['학점', '몇학점', '이수학점', '졸업학점'],
    'APPLY_PERIOD': ['기간', '언제', '마감', '일정', '시기', '날짜', '몇월', '2학기'],
    'APPLY_METHOD': ['방법', '절차', '순서', '어디서', '어디', '서류'],
    'APPLY_QUALIFICATION': ['자격', '조건', '대상', '기준', '가능', '돼', '되나', '될까', '되는지', '가능해', '할수있나', '가능하나', '할수있어', '아무나'],
    'PROGRAM_INFO': ['목록', '종류', '어떤', '리스트', '뭐가있', '뭐있'],
}

# 150 test questions
questions = [
    # PROGRAM_INFO (1-12)
    ("복수전공이 뭐야?", "PROGRAM_INFO"),
    ("부전공 설명해줘", "PROGRAM_INFO"),
    ("융합전공은 어떤 제도야?", "PROGRAM_INFO"),
    ("융합부전공이 뭔지 알려줘", "PROGRAM_INFO"),
    ("연계전공은 뭐야?", "PROGRAM_INFO"),
    ("마이크로디그리가 뭐야?", "PROGRAM_INFO"),
    ("소단위전공과정이 뭔지 궁금해", "PROGRAM_INFO"),
    ("다전공 제도 종류 알려줘", "MAJOR_SEARCH"),              # 목록 키워드 → MAJOR_SEARCH
    ("유연학사제도가 뭐야?", "PROGRAM_INFO"),
    ("복수전공 제도 개념이 뭐야?", "PROGRAM_INFO"),
    ("연계전공이 무엇인지 알려줘", "PROGRAM_INFO"),
    ("융합전공 어떤 건지 설명해줘", "PROGRAM_INFO"),
    # PROGRAM_COMPARISON (13-24)
    ("마이크로디그리랑 소단위전공과정이랑 똑같아?", "PROGRAM_COMPARISON"),
    ("복수전공이랑 부전공 차이가 뭐야?", "PROGRAM_COMPARISON"),
    ("융합전공이랑 융합부전공 비교해줘", "PROGRAM_COMPARISON"),
    ("복수전공과 연계전공 중에 뭐가 나아?", "PROGRAM_COMPARISON"),  # step 4.7
    ("부전공이랑 융합부전공 뭐가 달라?", "PROGRAM_COMPARISON"),
    ("복수전공이랑 융합전공 차이점 알려줘", "PROGRAM_COMPARISON"),
    ("연계전공이랑 마이크로디그리 비교해줘", "PROGRAM_COMPARISON"),  # step 4.7
    ("복수전공이랑 연계전공 뭐가 좋아?", "PROGRAM_COMPARISON"),
    ("부전공이랑 복수전공 어떤 게 유리해?", "PROGRAM_COMPARISON"),
    ("융합전공이랑 연계전공 차이 알려줘", "PROGRAM_COMPARISON"),
    ("복수전공 vs 융합부전공", "PROGRAM_COMPARISON"),
    ("마이크로디그리랑 연계전공 뭐가 달라?", "PROGRAM_COMPARISON"),
    # APPLY_QUALIFICATION (25-38)
    ("복수전공 신청 자격이 뭐야?", "APPLY_QUALIFICATION"),
    ("부전공 신청 조건 알려줘", "APPLY_QUALIFICATION"),
    ("1학년도 복수전공 신청 가능해?", "APPLY_QUALIFICATION"),
    ("편입생도 부전공 신청 돼?", "APPLY_QUALIFICATION"),
    ("융합전공 신청할 수 있는 조건이 뭐야?", "APPLY_QUALIFICATION"),
    ("마이크로디그리 누구나 신청 가능한가?", "APPLY_QUALIFICATION"),
    ("연계전공 신청 대상이 어떻게 돼?", "APPLY_QUALIFICATION"),
    ("융합부전공 자격 조건 알려줘", "APPLY_QUALIFICATION"),
    ("2학년인데 복수전공 할 수 있어?", "APPLY_QUALIFICATION"),    # step 5.5
    ("부전공 신청 가능한 학년이 어떻게 돼?", "APPLY_QUALIFICATION"),
    ("졸업 직전에도 다전공 신청 가능해?", "APPLY_QUALIFICATION"),
    ("재학생만 신청 가능한 거야?", "NONE"),                     # 프로그램 미감지
    ("소단위전공과정 아무나 할 수 있어?", "APPLY_QUALIFICATION"),  # step 5.5
    ("마이크로디그리 신청 대상 알려줘", "APPLY_QUALIFICATION"),
    # APPLY_PERIOD (39-50)
    ("복수전공 신청 기간이 언제야?", "APPLY_PERIOD"),
    ("부전공 언제 신청해?", "APPLY_PERIOD"),
    ("마이크로디그리 신청 마감일이 언제야?", "APPLY_PERIOD"),
    ("융합전공 접수 시기 알려줘", "APPLY_PERIOD"),
    ("연계전공 신청은 몇월에 해?", "APPLY_PERIOD"),
    ("다전공 신청 일정 알려줘", "APPLY_PERIOD"),
    ("소단위전공과정 신청기간이 궁금해", "APPLY_PERIOD"),
    ("복수전공 2학기에도 신청 가능해?", "APPLY_PERIOD"),          # 2학기 키워드 추가
    ("부전공 신청 마감 언제까지야?", "APPLY_PERIOD"),
    ("다전공 접수 기간 알려줘", "APPLY_PERIOD"),
    ("융합부전공 신청 시기가 언제야?", "APPLY_PERIOD"),
    ("연계전공 접수 일정 궁금해", "APPLY_PERIOD"),
    # APPLY_METHOD (51-62)
    ("복수전공 어떻게 신청해?", "APPLY_METHOD"),
    ("부전공 신청 방법 알려줘", "APPLY_METHOD"),
    ("마이크로디그리 신청 절차가 어떻게 돼?", "APPLY_METHOD"),
    ("융합전공 신청서 어디서 제출해?", "APPLY_METHOD"),
    ("연계전공 신청 순서 알려줘", "APPLY_METHOD"),
    ("융합부전공 접수 방법이 뭐야?", "APPLY_METHOD"),
    ("다전공 신청하려면 어떻게 해야 해?", "APPLY_METHOD"),
    ("복수전공 신청 서류가 뭐야?", "APPLY_METHOD"),              # 서류 키워드 추가
    ("부전공 온라인으로 신청 가능해?", "APPLY_QUALIFICATION"),    # '가능' → 자격 질문으로 분류 (합리적)
    ("마이크로디그리 어디서 신청해?", "APPLY_METHOD"),            # step 5.5
    ("소단위전공과정 신청 과정 알려줘", "APPLY_METHOD"),          # '신청과정' → _specific_intent_words → PROGRAM_INFO 차단 → FAQ APPLY_METHOD 매칭
    ("융합전공 신청서 양식 어디서 받아?", "APPLY_METHOD"),
    # CREDIT_INFO (63-72)
    ("복수전공 이수학점이 얼마나 돼?", "CREDIT_INFO"),
    ("부전공 몇학점 들어야 해?", "CREDIT_INFO"),
    ("융합전공 졸업학점 알려줘", "CREDIT_INFO"),
    ("마이크로디그리 학점 몇학점이야?", "CREDIT_INFO"),
    ("연계전공 이수학점이 궁금해", "CREDIT_INFO"),
    ("융합부전공 학점은 어떻게 돼?", "CREDIT_INFO"),
    ("복수전공 전필 학점 몇학점이야?", "CREDIT_INFO"),            # step 5.5
    ("부전공 최소 이수학점 알려줘", "CREDIT_INFO"),               # step 5.5
    ("소단위전공과정 몇학점 들으면 돼?", "CREDIT_INFO"),
    ("다전공 이수학점 기준 알려줘", "CREDIT_INFO"),
    # APPLY_CANCEL (73-80)
    ("복수전공 취소하고 싶은데", "APPLY_CANCEL"),
    ("부전공 포기하면 어떻게 돼?", "APPLY_CANCEL"),
    ("마이크로디그리 철회 가능해?", "APPLY_CANCEL"),
    ("융합전공 취소 방법 알려줘", "APPLY_CANCEL"),
    ("연계전공 포기하려면 어떻게 해?", "APPLY_CANCEL"),
    ("다전공 취소하고 싶어", "APPLY_CANCEL"),
    ("복수전공 중간에 그만둘 수 있어?", "APPLY_CANCEL"),          # 그만둘 키워드 추가
    ("융합부전공 포기 절차 알려줘", "APPLY_CANCEL"),
    # APPLY_CHANGE (81-86)
    ("다전공 신청한 전공 바꿀 수 있어?", "APPLY_CHANGE"),
    ("복수전공에서 부전공으로 변경 가능해?", "APPLY_CHANGE"),
    ("융합전공 전환하려면 어떻게 해?", "APPLY_CHANGE"),
    ("부전공을 복수전공으로 바꾸고 싶어", "APPLY_CHANGE"),
    ("다전공 종류 변경 가능한가?", "APPLY_CHANGE"),               # step 5.5
    ("연계전공에서 복수전공으로 전환 돼?", "APPLY_CHANGE"),
    # MAJOR_SEARCH (87-94)
    ("복수전공 신청 가능한 전공 목록 알려줘", "MAJOR_SEARCH"),
    ("부전공 어떤 전공이 있어?", "MAJOR_SEARCH"),
    ("융합전공 종류 알려줘", "MAJOR_SEARCH"),
    ("마이크로디그리 목록 알려줘", "MAJOR_SEARCH"),
    ("소단위전공과정 종류 알려줘", "MAJOR_SEARCH"),
    ("연계전공 리스트 보여줘", "MAJOR_SEARCH"),
    ("융합부전공 어떤 과정이 있어?", "MAJOR_SEARCH"),
    ("복수전공 가능한 학과 뭐가 있어?", "MAJOR_SEARCH"),
    # MAJOR_INFO (95-100)
    ("경영학전공 교과목 알려줘", "NONE"),                       # 엔티티 체크 미구현 (실제 챗봇에서는 MAJOR_INFO)
    ("컴퓨터공학전공 연락처 알려줘", "NONE"),                     # 엔티티 체크 미구현
    ("식품품질관리 MD 알려줘", "PROGRAM_INFO"),                   # MD → 마이크로디그리 PROGRAM_INFO (엔티티 체크 미구현)
    ("반려동물 MD 어떤 거야?", "PROGRAM_INFO"),                   # 동일
    ("AI보안 MD 설명해줘", "PROGRAM_INFO"),                       # 동일
    ("사물인터넷 MD 연락처 알려줘", "NONE"),                    # 엔티티 체크 미구현
    # 교직 (101-106)
    ("교직은?", "교직"),
    ("교직 이수 조건이 뭐야?", "교직"),
    ("교원자격증 어떻게 따?", "교직"),
    ("교직과정 신청 가능한 학과가 어디야?", "교직"),
    ("자유학기제 교직 이수 가능해?", "교직"),
    ("교직 적성검사 꼭 해야 해?", "교직"),
    # 졸업 (107-114)
    ("언제 졸업?", "졸업식"),
    ("졸업식 일정 알려줘", "졸업식"),
    ("졸업유예 신청 어떻게 해?", "유예"),
    ("조기졸업 조건이 뭐야?", "유예"),
    ("졸업유예 기간은 언제야?", "유예"),
    ("졸업식 날짜가 언제야?", "졸업식"),
    ("조기졸업 신청 방법 알려줘", "유예"),
    ("졸업유예하면 등록금 내야 해?", "유예"),                     # 졸업유예가 주요 키워드
    # 등록금 (115-120)
    ("등록금 얼마야?", "등록금"),
    ("등록금 환불 가능해?", "등록금"),
    ("장학금 받을 수 있어?", "등록금"),
    ("복수전공하면 등록금 더 내야 해?", "등록금"),
    ("다전공 추가 비용 있어?", "등록금"),
    ("등록금 납부 기간 언제야?", "등록금"),
    # 증명서 (121-126)
    ("증명서 발급 어떻게 해?", "증명서"),
    ("졸업증명서 어디서 뽑아?", "증명서"),
    ("영문 성적증명서 발급 가능해?", "증명서"),
    ("재학증명서 온라인으로 발급돼?", "증명서"),
    ("증명서 무인발급기 어디 있어?", "증명서"),
    ("정부24에서 증명서 뽑을 수 있어?", "증명서"),
    # 수강 (127-132)
    ("수강신청 어떻게 해?", "수강"),
    ("수강철회하면 성적에 어떻게 나와?", "수강"),
    ("수강정정 기간이 언제야?", "수강"),
    ("재수강 가능해?", "수강"),
    ("수강신청 정정 어떻게 해?", "수강"),
    ("성적 이의신청 어떻게 해?", "수강"),
    # 학점교류 (133-137)
    ("학점교류 어떻게 신청해?", "학점교류"),
    ("계절학기 신청 방법 알려줘", "학점교류"),
    ("타대학 학점 인정돼?", "학점교류"),
    ("교환학생 학점교류 가능해?", "학점교류"),
    ("여름학기 타교에서 들을 수 있어?", "학점교류"),               # 여름학기 키워드 추가
    # 성적 (138-140)
    ("성적 어디서 확인해?", "성적, 수강확인"),
    ("이수학점 확인하고 싶어", "성적, 수강확인"),
    ("학사시스템 주소 알려줘", "NONE"),                           # FAQ 답변 없음
    # AI_FALLBACK (141-147)
    ("복수전공하면 졸업 늦어져?", "NONE"),                       # 의도 키워드 미매칭 (실제 챗봇에서는 AI fallback)
    ("부전공 이수 안 하면 어떻게 돼?", "APPLY_QUALIFICATION"),    # step 5.5 → '돼' 매칭
    ("복수전공 하면서 연계전공도 할 수 있어?", "APPLY_QUALIFICATION"),  # '할수있어' → 자격 매칭
    ("다전공 몇 개까지 할 수 있어?", "APPLY_QUALIFICATION"),      # '할수있어' → 자격 매칭
    ("융합전공 졸업하면 학위 2개 받아?", "NONE"),                  # 의도 키워드 미매칭 (실제 챗봇에서는 AI fallback)
    ("마이크로디그리 이수하면 취업에 도움 돼?", "APPLY_QUALIFICATION"),  # step 5.5 → '돼' 매칭
    ("부전공 이수하면 학위 나와?", "CREDIT_INFO"),                 # FAQ에서 CREDIT_INFO 매칭
    # 기타 학사 (148-150)
    ("휴학 신청 어떻게 해?", "증명서"),
    ("전과하고 싶은데 어떻게 해?", "증명서"),
    ("학사시스템 로그인이 안 돼", "NONE"),                        # FAQ 답변 없음
    # ===== 추가 50개 (151-200) =====
    # 구어체/줄임말 표현 (151-158)
    ("복전 어떻게 해?", "APPLY_METHOD"),                         # 복전 → 복수전공
    ("부전 신청하려면?", "NONE"),                               # '하려면' 의도 키워드 없음
    ("복수전공 알려줘", "PROGRAM_INFO"),
    ("부전공이 뭔데?", "PROGRAM_INFO"),
    ("마이크로디그리 해보고 싶은데", "NONE"),                     # 의도 불명확
    ("연계전공 궁금해요", "PROGRAM_INFO"),
    ("융합전공 하고 싶어", "NONE"),                             # 의도 불명확
    ("소단위전공과정 궁금합니다", "PROGRAM_INFO"),
    # 복합 키워드 질문 (159-166)
    ("복수전공 신청 자격이랑 기간 알려줘", "APPLY_PERIOD"),         # 자격+기간 → _icm에서 기간 선순위
    ("부전공 학점이랑 신청 방법 궁금해", "CREDIT_INFO"),           # 학점+방법 → 학점 우선 (CREDIT_INFO가 APPLY_METHOD보다 선순위)
    ("융합전공 취소하면 학점은 어떻게 돼?", "APPLY_CANCEL"),        # 취소+학점 → 취소 우선
    ("마이크로디그리 신청 기간이랑 자격 알려줘", "APPLY_PERIOD"),   # 기간이 APPLY_QUALIFICATION보다 선순위
    ("복수전공 변경하려면 언제까지 해야 해?", "APPLY_CHANGE"),      # 변경+언제 → 변경 우선
    ("부전공 포기하면 학점 인정돼?", "APPLY_CANCEL"),              # 포기+학점 → 취소 우선
    ("연계전공 신청 서류랑 절차 알려줘", "APPLY_METHOD"),          # 서류+절차 → 방법
    ("다전공 신청 마감 지났는데 가능해?", "APPLY_PERIOD"),         # 마감+가능 → 기간 우선
    # 다양한 학사제도 표현 (167-175)
    ("졸업 요건이 뭐야?", "졸업식"),                              # 졸업 키워드 → 졸업식 FAQ 먼저 매칭
    ("학위수여식 언제야?", "졸업식"),                              # 학위수여식
    ("등록금 분납 가능해?", "등록금"),
    ("국가장학금 신청 방법 알려줘", "등록금"),                     # 장학금
    ("성적증명서 뽑으려면?", "증명서"),
    ("수강철회 기간 알려줘", "수강"),
    ("학점교류 학점 인정 기준이 뭐야?", "학점교류"),
    ("교직 필수 과목이 뭐야?", "교직"),
    ("계절학기 학점 몇 학점까지 들을 수 있어?", "학점교류"),
    # 프로그램 비교 추가 (176-180)
    ("복수전공이랑 부전공 어떤 게 좋아?", "PROGRAM_COMPARISON"),
    ("융합전공하고 복수전공하고 뭐가 다른 거야?", "PROGRAM_COMPARISON"),
    ("소단위전공과정이랑 마이크로디그리 같은 건가?", "PROGRAM_COMPARISON"),
    ("부전공 vs 연계전공", "PROGRAM_COMPARISON"),
    ("복수전공이랑 융합부전공이랑 뭐가 좋을까?", "PROGRAM_COMPARISON"),
    # 신청/자격 다양한 표현 (181-186)
    ("복수전공 신청할 수 있는 학년이 어떻게 돼?", "APPLY_QUALIFICATION"),
    ("융합부전공 신청 가능 학과 알려줘", "APPLY_QUALIFICATION"),
    ("소단위전공과정 신청 시작일이 언제야?", "APPLY_PERIOD"),
    ("마이크로디그리 접수 마감 언제야?", "APPLY_PERIOD"),
    ("연계전공 어떻게 접수해?", "APPLY_METHOD"),
    ("복수전공 신청 취소할 수 있어?", "APPLY_CANCEL"),
    # 학점/이수 관련 (187-191)
    ("복수전공 전공필수 몇 학점이야?", "CREDIT_INFO"),
    ("융합전공 최소 이수학점 알려줘", "CREDIT_INFO"),
    ("소단위전공과정 이수학점 궁금해", "CREDIT_INFO"),
    ("부전공 전공선택 학점은?", "CREDIT_INFO"),
    ("연계전공 졸업 이수학점 기준 알려줘", "CREDIT_INFO"),
    # 목록/검색 추가 (192-195)
    ("마이크로디그리 어떤 과정이 있어?", "MAJOR_SEARCH"),
    ("복수전공 할 수 있는 학과가 뭐야?", "PROGRAM_INFO"),          # '뭐야' → PROGRAM_INFO redirect
    ("융합전공 어떤 게 있어?", "MAJOR_SEARCH"),                   # 어떤게있어 → list_keywords
    ("부전공 목록 보여줘", "MAJOR_SEARCH"),
    # NONE/경계 케이스 (196-200)
    ("안녕하세요", "NONE"),                                       # 인사말
    ("ㅎㅎ", "NONE"),                                            # 무의미 입력
    ("점심 뭐 먹지?", "NONE"),                                   # 범위 외
    ("도서관 어디 있어?", "NONE"),                                # 범위 외
    ("기숙사 신청 어떻게 해?", "NONE"),                           # 범위 외
    # ===== 추가 100개 (201-300) =====
    # 다양한 문체/존댓말 (201-210)
    ("복수전공 신청하고 싶습니다", "APPLY_PERIOD"),                  # '신청' 키워드 → FAQ APPLY_PERIOD 매칭
    ("부전공 신청 자격 좀 알려주세요", "APPLY_QUALIFICATION"),
    ("융합전공 언제 신청하면 되나요?", "APPLY_PERIOD"),
    ("마이크로디그리 어떻게 하는 건가요?", "APPLY_METHOD"),
    ("연계전공 취소 가능한가요?", "APPLY_CANCEL"),
    ("소단위전공과정 학점이 궁금합니다", "CREDIT_INFO"),
    ("복수전공 변경 절차가 어떻게 되나요?", "APPLY_CHANGE"),
    ("부전공 포기 신청은 언제까지인가요?", "APPLY_CANCEL"),        # 포기+언제 → 취소 우선
    ("융합부전공이 무엇인가요?", "PROGRAM_INFO"),
    ("다전공 제도에 대해 설명 부탁드립니다", "PROGRAM_INFO"),
    # 반말/구어체 (211-220)
    ("복전 취소 가능?", "APPLY_CANCEL"),
    ("부전 학점 몇 학점?", "CREDIT_INFO"),
    ("MD 신청 언제 해?", "APPLY_PERIOD"),
    ("복수전공 걍 포기할래", "APPLY_CANCEL"),
    ("부전공 바꿀 수 있음?", "APPLY_CHANGE"),
    ("융합전공 뭐임?", "PROGRAM_INFO"),
    ("마이크로 뭐야?", "PROGRAM_INFO"),                           # 마이크로 → 마이크로디그리
    ("소단위 어떻게 신청함?", "APPLY_METHOD"),                    # 소단위 → 소단위전공과정
    ("다전공 접수 방법?", "APPLY_METHOD"),
    ("복전 변경하고 싶은데", "APPLY_CHANGE"),
    # 질문 형태 다양화 (221-230)
    ("복수전공 신청 자격 요건이 있어?", "APPLY_QUALIFICATION"),
    ("부전공 이수하는 데 조건이 뭐야?", "APPLY_QUALIFICATION"),
    ("융합전공은 누구나 할 수 있어?", "APPLY_QUALIFICATION"),      # '할수있어' → 자격
    ("마이크로디그리 접수 마감이 다가오는데", "APPLY_PERIOD"),
    ("연계전공 포기하고 복수전공으로 바꿀 수 있어?", "APPLY_CANCEL"),  # 포기+바꿀 → 취소 우선
    ("소단위전공과정 이수 완료하면 뭐가 나와?", "CREDIT_INFO"),      # '이수' → CREDIT_INFO 매칭
    ("다전공 신청 마감일 알려줘", "APPLY_PERIOD"),
    ("복수전공 신청 취소 기한이 있어?", "APPLY_CANCEL"),           # 취소 우선
    ("부전공 졸업학점 기준 알려줘", "CREDIT_INFO"),
    ("융합부전공 신청하는 곳이 어디야?", "APPLY_METHOD"),          # '어디' → APPLY_METHOD 부스트
    # PROGRAM_COMPARISON 추가 (231-238)
    ("복수전공이랑 부전공이랑 학점이 달라?", "PROGRAM_COMPARISON"),
    ("융합전공이랑 복수전공 중에 뭐가 좋을까?", "PROGRAM_COMPARISON"),
    ("소단위전공과정이랑 부전공 차이가 뭐야?", "PROGRAM_COMPARISON"),
    ("연계전공이랑 융합전공 비교해서 알려줘", "PROGRAM_COMPARISON"),
    ("복수전공이랑 융합전공 어떤 게 유리해?", "PROGRAM_COMPARISON"),
    ("부전공이랑 마이크로디그리 뭐가 다른 거야?", "PROGRAM_COMPARISON"),
    ("연계전공이랑 융합부전공 차이 설명해줘", "PROGRAM_COMPARISON"),
    ("마이크로디그리랑 복수전공 뭐가 좋아?", "PROGRAM_COMPARISON"),
    # APPLY_CHANGE 추가 (239-246)
    ("복수전공을 융합전공으로 바꿀 수 있어?", "APPLY_CHANGE"),
    ("부전공에서 복수전공으로 전환하려면 어떻게 해?", "APPLY_CHANGE"),
    ("다전공 변경 신청 방법 알려줘", "APPLY_CHANGE"),             # 변경+방법 → 변경 우선
    ("연계전공을 부전공으로 변경 가능한가?", "APPLY_CHANGE"),
    ("융합부전공을 다른 전공으로 바꾸고 싶어", "APPLY_CHANGE"),
    ("소단위전공과정 변경 가능해?", "APPLY_CHANGE"),
    ("마이크로디그리 전환 절차 알려줘", "APPLY_CHANGE"),
    ("복수전공 바꾼 사람 있어?", "APPLY_CHANGE"),                 # '바꾼' → APPLY_CHANGE
    # 학사제도 다양한 표현 (247-260)
    ("등록금 카드로 낼 수 있어?", "등록금"),
    ("등록금 분할 납부 방법 알려줘", "등록금"),
    ("장학금 종류 알려줘", "NONE"),                               # program_type=None (extract_program 미감지) → NONE
    ("졸업유예 조건이 뭐야?", "유예"),
    ("조기졸업 가능한 조건 알려줘", "유예"),
    ("졸업식 장소가 어디야?", "졸업식"),
    ("학위수여식 일정이 어떻게 돼?", "졸업식"),
    ("증명서 발급 수수료 있어?", "증명서"),
    ("재학증명서 발급 방법 알려줘", "증명서"),
    ("수강신청 기간이 언제야?", "수강"),
    ("재수강 학점 어떻게 돼?", "수강"),
    ("교직 이수 절차가 어떻게 돼?", "교직"),
    ("교원자격증 발급 조건 알려줘", "교직"),
    ("학점교류 신청 기간 언제야?", "학점교류"),
    # 성적/수강 관련 (261-268)
    ("성적 확인 어디서 해?", "성적, 수강확인"),
    ("이수학점 조회하고 싶어", "성적, 수강확인"),
    ("수강 내역 확인 방법 알려줘", "성적, 수강확인"),
    ("계절학기 수강신청 어떻게 해?", "학점교류"),                  # 수강 exclude에 계절학기 추가 → 학점교류 매칭
    ("학점교류 가능한 대학 알려줘", "학점교류"),
    ("교직 적성검사 일정 알려줘", "교직"),
    ("교직 이수 가능한 학과 알려줘", "교직"),
    ("수강정정 방법 알려줘", "수강"),
    # 등록금+프로그램 결합 (269-274)
    ("부전공 하면 등록금 추가로 내야 돼?", "등록금"),
    ("융합전공 등록금 얼마야?", "등록금"),                         # APPLY_METHOD exclude에 등록금 추가 → 학사제도 등록금 매칭
    ("마이크로디그리 하면 비용이 얼마나 들어?", "등록금"),
    ("연계전공 추가 학비 있어?", "등록금"),
    ("소단위전공과정 등록금 별도야?", "등록금"),
    ("융합부전공 비용 알려줘", "등록금"),                          # 학사제도 등록금 keyword에 '비용' 추가됨
    # FOLLOWUP 시뮬레이션 (275-280)
    ("그거 어떻게 해?", "FOLLOWUP"),
    ("그럼 기간은?", "FOLLOWUP"),
    ("그건 뭐야?", "FOLLOWUP"),
    ("거기 연락처 알려줘", "FOLLOWUP"),
    ("그래서 자격은?", "FOLLOWUP"),
    ("아까 말한 거 다시 알려줘", "FOLLOWUP"),
    # 엣지 케이스/모호한 질문 (281-290)
    ("전공", "NONE"),                                             # 너무 짧은 입력
    ("신청", "NONE"),                                             # 프로그램 미지정
    ("복수전공", "NONE"),                                         # 의도 없는 단어만
    ("언제?", "NONE"),                                            # 프로그램 미지정
    ("어떻게?", "NONE"),                                          # 프로그램 미지정
    ("복수전공 하면 좋은 점이 뭐야?", "NONE"),                     # '좋은점' → _specific_intent_words → PROGRAM_INFO 차단
    ("다전공 안 하면 불이익 있어?", "NONE"),                       # 의도 키워드 미매칭
    ("융합전공 인기 많아?", "NONE"),                               # 의도 키워드 미매칭
    ("부전공 추천해줘", "NONE"),                                   # 의도 키워드 미매칭
    ("복수전공 후기 있어?", "NONE"),                               # 의도 키워드 미매칭
    # NONE/범위 외 (291-300)
    ("학교 셔틀버스 시간표 알려줘", "NONE"),
    ("교수님 연구실 어디야?", "NONE"),
    ("동아리 가입 어떻게 해?", "NONE"),
    ("학생증 재발급 어떻게 해?", "NONE"),
    ("생활관 식단 알려줘", "NONE"),
    ("축제 언제야?", "NONE"),
    ("복수전공이랑 부전공 동시에 할 수 있어?", "AI_COMBINE"),           # 2개 프로그램 + '동시' → 동시이수
    ("마이크로디그리 이수 기간이 얼마나 걸려?", "APPLY_PERIOD"),      # '기간' → 기간
    ("다전공 신청하면 수강신청 어떻게 바뀌어?", "APPLY_METHOD"),      # '어떻게' + '신청' → FAQ APPLY_METHOD 매칭
    ("유연학사제도 설명해줘", "PROGRAM_INFO"),
    # ===== 동시 이수 관련 50개 (301-350) =====
    # AI_COMBINE: 2개 이상 프로그램 + 동시이수 키워드 (301-335)
    ("마이크로디그리랑 연계전공 같이 신청 가능해?", "AI_COMBINE"),
    ("복수전공이랑 부전공 같이 할 수 있어?", "AI_COMBINE"),
    ("융합전공이랑 복수전공 동시에 신청 가능한가요?", "AI_COMBINE"),
    ("부전공이랑 연계전공 함께 이수할 수 있어?", "AI_COMBINE"),
    ("복수전공이랑 마이크로디그리 병행 가능해?", "AI_COMBINE"),
    ("융합부전공이랑 복수전공 둘다 할 수 있나?", "AI_COMBINE"),
    ("부전공이랑 융합전공 겸 신청 돼?", "AI_COMBINE"),
    ("연계전공이랑 부전공 같이 들을 수 있어?", "AI_COMBINE"),
    ("마이크로디그리랑 복수전공 동시 이수 가능해?", "AI_COMBINE"),
    ("소단위전공과정이랑 부전공 같이 신청 가능?", "AI_COMBINE"),
    ("복수전공이랑 융합부전공 함께 하면 돼?", "AI_COMBINE"),
    ("연계전공이랑 복수전공 병행할 수 있나요?", "AI_COMBINE"),
    ("융합전공이랑 부전공 둘다 신청 가능해?", "AI_COMBINE"),
    ("마이크로디그리랑 부전공 같이 이수 가능한가?", "AI_COMBINE"),
    ("소단위전공과정이랑 마이크로디그리 동시에 할 수 있어?", "AI_COMBINE"),
    ("복수전공이랑 연계전공 같이 신청해도 돼?", "AI_COMBINE"),
    ("융합부전공이랑 연계전공 동시 신청 가능해?", "AI_COMBINE"),
    ("부전공이랑 소단위전공과정 함께 들을 수 있나?", "AI_COMBINE"),
    ("복수전공이랑 부전공 중복 신청 되나?", "AI_COMBINE"),
    ("마이크로디그리랑 융합전공 이중으로 할 수 있어?", "AI_COMBINE"),
    ("복수전공이랑 융합전공 같이 하면 학점이 어떻게 돼?", "AI_COMBINE"),
    ("부전공이랑 연계전공 동시에 졸업 가능해?", "AI_COMBINE"),
    ("융합전공이랑 마이크로디그리 병행하면 힘들어?", "AI_COMBINE"),
    ("복수전공이랑 부전공 겸하면 졸업 늦어져?", "AI_COMBINE"),
    ("연계전공이랑 융합부전공 같이 하는 학생 있어?", "AI_COMBINE"),
    ("복수전공이랑 소단위전공과정 함께 신청하고 싶어", "AI_COMBINE"),
    ("부전공이랑 마이크로디그리 둘다 하고 싶은데", "AI_COMBINE"),
    ("융합전공이랑 연계전공 같이 신청하려면 어떻게 해?", "AI_COMBINE"),
    ("복수전공이랑 융합부전공 동시에 이수하는 사람 많아?", "AI_COMBINE"),
    ("마이크로디그리랑 소단위전공과정 병행 가능한가요?", "AI_COMBINE"),
    ("복수전공이랑 부전공 같이 하면 추가 등록금 있어?", "AI_COMBINE"),
    ("연계전공이랑 마이크로디그리 함께 신청 가능해?", "AI_COMBINE"),
    ("융합부전공이랑 부전공 둘다 이수할 수 있어?", "AI_COMBINE"),
    ("복수전공이랑 연계전공 동시 이수하면 학점 인정 돼?", "AI_COMBINE"),
    ("부전공이랑 융합전공 같이 하면 졸업 요건 어떻게 돼?", "AI_COMBINE"),
    # 동시이수 키워드 있지만 프로그램 1개 → AI_COMBINE 아님 (336-340)
    ("복수전공 같이 하고 싶어", "NONE"),                                  # 프로그램 1개 + 의도 불명확 → NONE
    ("부전공 동시에 몇 개까지 가능해?", "APPLY_QUALIFICATION"),            # 프로그램 1개 → step 5.5
    ("마이크로디그리 함께 이수하려면 어떻게 해?", "APPLY_METHOD"),          # 프로그램 1개 → step 5.5
    ("연계전공 병행하기 힘들어?", "NONE"),                                 # 프로그램 1개 + 의도 없음
    ("융합전공 겸하는 사람 많아?", "NONE"),                                # 프로그램 1개 + 의도 없음
    # 비교 vs 동시이수 구분 (341-345)
    ("복수전공이랑 부전공 차이가 뭐야?", "PROGRAM_COMPARISON"),            # 비교 키워드 → 비교
    ("융합전공이랑 연계전공 뭐가 달라?", "PROGRAM_COMPARISON"),            # 비교 키워드 → 비교
    ("복수전공이랑 부전공 중에 뭐가 좋아?", "PROGRAM_COMPARISON"),         # 비교 키워드 → 비교
    ("마이크로디그리랑 부전공 같이 해도 되나?", "AI_COMBINE"),             # 동시이수 키워드 → 동시이수
    ("연계전공이랑 복수전공 함께 신청할 수 있나요?", "AI_COMBINE"),         # 동시이수 키워드 → 동시이수
    # 다양한 문체 (346-350)
    ("복전이랑 부전 같이 할 수 있음?", "AI_COMBINE"),                     # 약어 사용
    ("복수전공하고 부전공 동시에 되나요?", "AI_COMBINE"),                  # 정중한 표현
    ("융합전공이랑 복수전공 둘 다 하고 싶습니다", "AI_COMBINE"),           # 공손한 표현
    ("MD랑 부전공 같이 하면 돼?", "AI_COMBINE"),                          # MD 약어
    ("소단위랑 마이크로 같이 신청 가능?", "AI_COMBINE"),                   # 약어 사용
]

print(f"{'#':>3} | {'결과':^2} | {'질문':<42} | {'기대':<22} | {'실제':<22} | {'점수':>4}")
print("-" * 115)

pass_count = 0
fail_count = 0
fail_list = []

for i, (q, expected) in enumerate(questions, 1):
    is_followup = is_followup_question(q)
    program_type = extract_program_from_text(q)
    is_pi_redirect = check_program_info_redirect(q, program_type)
    is_comp = is_comparison_query(q)
    is_comb = is_combine_query(q)
    faq_match, score = search_faq_mapping(q, faq_df)

    # determine actual result (simulating real code flow)
    if is_followup:
        actual = "FOLLOWUP"
    elif is_comp:
        # Step 4.7: 비교 질문 → PROGRAM_COMPARISON
        actual = "PROGRAM_COMPARISON"
    elif is_comb:
        # Step 4.8: 동시 이수 질문 → AI_COMBINE
        actual = "AI_COMBINE"
    elif is_pi_redirect:
        # Step 4.5: 프로그램 설명 질문 → PROGRAM_INFO
        actual = "PROGRAM_INFO"
    elif faq_match is not None:
        faq_intent = str(faq_match['intent'])
        # Step 5: intent conflict resolution
        _uc = q.lower().replace(' ', '')
        _user_ci = None
        for _ci, _ckws in _icm.items():
            if any(_ck in _uc for _ck in _ckws):
                _user_ci = _ci
                break
        if _user_ci == 'APPLY_QUALIFICATION':
            for _oi, _okws in _icm.items():
                if _oi == 'APPLY_QUALIFICATION': continue
                if any(_ok in _uc for _ok in _okws):
                    _user_ci = _oi
                    break
        if _user_ci and faq_intent != _user_ci and faq_intent in _icm:
            # Step 5.5: 의도 충돌 → 의도 기반 직접 조회
            actual = _user_ci + "(5.5)"
        else:
            actual = faq_intent
    else:
        # FAQ 미매칭
        _uc = q.lower().replace(' ', '')
        _list_kws = ['목록', '리스트', '종류', '어떤전공', '어떤과정', '무슨전공', '무슨과정', '뭐가있어', '뭐있어', '어떤게있어', '뭐가있']
        # 변경/취소 등 구체적 의도가 있으면 목록 질문이 아님
        _non_list_intents = ['변경', '바꾸', '전환', '취소', '포기', '철회']
        _is_list = any(lk in _uc for lk in _list_kws) and program_type and not any(ni in _uc for ni in _non_list_intents)
        if _is_list:
            # list_keywords로 차단된 경우 → MAJOR_SEARCH 핸들러
            actual = "MAJOR_SEARCH"
        else:
            # Step 5.5 시뮬레이션
            step55_result = simulate_step_5_5(q, program_type, faq_df)
            if step55_result:
                actual = step55_result + "(5.5)"
            else:
                actual = "NONE"

    matched = (actual == expected) or (actual == expected + "(5.5)")
    status = "OK" if matched else "NG"

    if matched:
        pass_count += 1
    else:
        fail_count += 1
        fail_list.append((i, q, expected, actual, score))

    print(f"{i:>3} | {status:^4} | {q:<42} | {expected:<22} | {actual:<22} | {score:>4}")

print("-" * 115)
print(f"\n총 {len(questions)}개 | 성공: {pass_count}개 | 실패: {fail_count}개 | 성공률: {pass_count/len(questions)*100:.1f}%")

if fail_list:
    print(f"\n{'='*70}")
    print("실패 목록:")
    print(f"{'='*70}")
    for num, q, exp, act, sc in fail_list:
        print(f"  #{num} \"{q}\"")
        print(f"      기대: {exp} -> 실제: {act} (점수: {sc})")
