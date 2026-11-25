"""증상 분석용 규칙 (LLM 결과를 보정하는 안전장치).

1. 증상 규칙(RULES): 문장에 특정 표현이 있으면 E-Gen 질환 코드를 추가하고 KTAS 상한을 건다.
2. 비급성 표지(NON_ACUTE): "예전에", "한 달 됐어요"처럼 급성이 아닌 맥락이면 KTAS 2 규칙을 끈다.
3. 증상 어휘(SYMPTOM): 증상 단어가 하나도 없으면 분류하지 않고 되묻는다.
4. 정합성 필터: 나이·문맥과 맞지 않는 질환 코드를 지운다 (예: 70세에게 영유아 코드).

규칙은 '덜 위급하게 판단하는 실수(Under-triage)'를 막는 방향으로만 작동한다.
KTAS를 낮추는 규칙은 없다.
"""
import re
from dataclasses import dataclass, field
from typing import Callable

Age = int | None


def child(age: Age, limit: int = 6) -> bool:
    return age is not None and age <= limit


def adult(age: Age, limit: int = 6) -> bool:
    return age is None or age > limit


@dataclass
class Rule:
    pattern: str
    code: str | None = None          # 추가할 E-Gen 질환 코드
    ceiling: int | None = None       # 허용 최대 KTAS (작을수록 위급)
    when: Callable[[str, Age], bool] | None = None   # 추가 조건
    acute_only: bool = True          # 비급성 표지가 있으면 끔 (KTAS 1 규칙은 False)
    negatable: bool = True           # 걸린 구간에 '안/않/없'이 있으면 무시 (표현 자체에 '안'이 들어가면 False)
    regex: re.Pattern = field(init=False)

    def __post_init__(self):
        self.regex = re.compile(self.pattern)


def _weeks(text: str) -> int | None:
    """'임신 30주', '34주라' 같은 표현에서 주수를 꺼낸다"""
    m = re.search(r"(\d{1,2})\s*주", text)
    return int(m.group(1)) if m else None


def _ped(code_child: str, code_adult: str) -> Callable[[str, Age], str]:
    return lambda text, age: code_child if child(age, 12) else code_adult


# ── 증상 규칙 ─────────────────────────────────────────────────────────
# code가 함수면 (text, age)로 코드를 고른다
RULES: list[Rule] = [
    # KTAS 1: 소생 (비급성 표지와 무관하게 항상 적용)
    Rule(r"의식(이|을)?\s*(없|잃)|반응(이)?\s*없|숨(을)?\s*(안|못)\s*쉬|호흡(이)?\s*없|심정지|맥(이)?\s*(없|안\s*잡)", ceiling=1, acute_only=False),
    Rule(r"깨워도\s*(안\s*일어|일어나지)|입술(이)?\s*(파래|파랗)|청색증|축\s*처져서?\s*반응", ceiling=1, acute_only=False),
    Rule(r"경련(을)?\s*\d+\s*분\s*째|경련이?\s*(계속|안\s*멈)", ceiling=1, acute_only=False),
    Rule(r"피가?\s*뿜|찔렸|총에|\d+\s*층.{0,8}떨어|높은\s*곳.{0,8}떨어", ceiling=1, acute_only=False),
    Rule(r"(목에|기도에)\s*걸려.{0,15}(숨|호흡)", code="Y0091", ceiling=1, acute_only=False, when=lambda t, a: adult(a)),
    Rule(r"(교통)?사고.{0,30}창백|창백.{0,30}사고", ceiling=1, acute_only=False, when=lambda t, a: adult(a, 12)),

    # 심혈관
    Rule(r"가슴.{0,8}(아프|아파|아푸|통증|조이|조여|쥐어짜|답답|묵직|막힌|막혀)|흉통", code="Y0010", ceiling=2),
    Rule(r"(명치|윗배).{0,25}(식은땀|땀을\s*많이|땀이)", code="Y0010", ceiling=2),
    Rule(r"(가슴|등).{0,10}(찢어|칼로\s*찌르)|(찢어|칼로\s*찌르).{0,10}등", code="Y0041", ceiling=2),
    Rule(r"허리.{0,10}찢어|찢어.{0,10}허리|배에서\s*.{0,4}(맥박|뛰)", code="Y0042", ceiling=2),
    Rule(r"심장이?\s*(너무\s*)?빨리|맥박이?\s*.{0,6}(느려|빨라|이상)", ceiling=2),

    # 뇌졸중·두통·두부외상
    Rule(r"말(이)?\s*어눌|발음(이)?\s*(이상|어눌|꼬)|말을\s*(못\s*알아|이상하게)|엉뚱한\s*말"
         r"|(팔|다리|손발)(에|가|을|를)?\s*(힘(이)?\s*(없|빠)|못\s*쓰)|한쪽.{0,6}(마비|저림|처짐|눈|팔|다리|손발)"
         r"|편마비|얼굴.{0,6}(처지|비뚤)|두\s*개로\s*보|겹쳐\s*보|걸을\s*때.{0,8}쏠|젓가락을.{0,6}떨어뜨",
         code=lambda t, a: None if a is not None and a < 15 else "Y0020", ceiling=2),
    Rule(r"머리(가)?\s*깨질|벼락\s*두통|망치로\s*맞은|평생.{0,10}(처음|이런).{0,6}두통|극심한\s*두통|두통.{0,20}목이?\s*뻣뻣",
         code="Y0031", ceiling=2),
    Rule(r"머리.{0,8}(부딪|박|다쳤|다치)", code="Y0032"),
    Rule(r"머리.{0,8}(부딪|박|다쳤|다치).{0,30}(토|졸|의식|횡설|피가|멍해)|(혈전약|항응고|와파린|아스피린).{0,25}머리",
         code="Y0032", ceiling=2),
    Rule(r"횡설수설|정신이?\s*(오락가락|없|멍)|의식(이)?\s*(흐려|흐릿|왔다갔다)", ceiling=2),

    # 소화기
    Rule(r"피를?\s*(한\s*바가지\s*)?토|토혈|커피색.{0,6}토", code="Y0081", ceiling=2,
         when=lambda t, a: adult(a)),
    Rule(r"검은\s*변|짜장\s*(같은|색)|변기에.{0,8}피|혈변|피가\s*섞인\s*변", code="Y0081",
         ceiling=3, when=lambda t, a: adult(a)),
    Rule(r"(검은\s*변|짜장\s*(같은|색)|변기에.{0,8}피|혈변).{0,30}(어지|기운|쓰러|창백|많이)|(어지|기운).{0,30}(검은\s*변|짜장|혈변)",
         code="Y0081", ceiling=2, when=lambda t, a: adult(a)),
    Rule(r"(노래|노랗|황달|콜라색)", code="Y0052", ceiling=3),
    Rule(r"(노래|노랗|황달|콜라색).{0,30}(열|오한)|(열|오한).{0,30}(노래|노랗|황달|콜라색)", code="Y0052", ceiling=2),
    Rule(r"오른쪽\s*(윗배|갈비뼈\s*아래)", code="Y0051", ceiling=3),
    Rule(r"배가?\s*딱딱|만지지도\s*못|오른쪽\s*아랫배|아래로\s*내려왔|방귀가?\s*안\s*나", code="Y0060", ceiling=3, negatable=False,
         when=lambda t, a: not re.search(r"임신|산모", t)),
    Rule(r"배가?\s*딱딱.{0,15}만지지", code="Y0060", ceiling=2),
    Rule(r"자지러지|딸기잼|배가?\s*부풀", code="Y0070", ceiling=2, when=lambda t, a: child(a, 3)),

    # 이물·중독 (영유아)
    Rule(r"삼켰|삼킨|마셨", code="Y0082", ceiling=3, when=lambda t, a: child(a)),
    Rule(r"건전지|자석|침을\s*(계속\s*)?흘|세제|락스|약을.{0,6}(먹|삼)", code="Y0082", ceiling=2,
         when=lambda t, a: child(a) and re.search(r"삼켰|삼킨|마셨|먹었", t)),
    Rule(r"(사레|땅콩|콩|조각).{0,30}(기침|쌕쌕|숨)", code="Y0092", ceiling=3, when=lambda t, a: child(a)),
    Rule(r"(사레|땅콩|콩|조각).{0,30}(쌕쌕|숨소리|숨쉬)", code="Y0092", ceiling=2, when=lambda t, a: child(a)),

    # 호흡기
    Rule(r"피를?\s*기침|기침.{0,12}피|객혈|각혈", code="Y0091", ceiling=3, when=lambda t, a: adult(a)),
    Rule(r"(기침.{0,12}피|객혈|각혈).{0,15}(컵|많이|바가지)|(컵|바가지).{0,10}피", code="Y0091", ceiling=2,
         when=lambda t, a: adult(a) and re.search(r"기침|객혈|각혈", t)),
    Rule(r"숨(이)?\s*(차|가빠|가쁘|막히)|호흡\s*곤란|숨쉬기\s*(힘들|어렵)|숨을\s*(헐떡|가쁘게)|갈비뼈\s*사이.{0,6}들어", ceiling=2),

    # 신경
    Rule(r"경련|발작", ceiling=2),
    Rule(r"목이?\s*뻣뻣.{0,15}(열|빛)|(열|\d+\s*도).{0,15}목이?\s*뻣뻣", ceiling=2),

    # 영아·소아
    Rule(r"생후\s*([1-9]|1[0-2])\s*주|생후\s*[12]\s*개월|신생아", ceiling=2,
         when=lambda t, a: re.search(r"열|처져|젖을\s*안|안\s*먹", t)),
    Rule(r"눌러도\s*(안\s*)?(없어지|사라지)", ceiling=2, negatable=False),
    Rule(r"미숙아|이른둥이|(낳았|출산).{0,20}(작아|주라)", code="Y0100", ceiling=2),

    # 산과·부인과
    Rule(r"(임신|산모|\d+\s*주).{0,30}(피|출혈|딱딱|번쩍|부었|두통|머리가)", code="Y0112",
         ceiling=2, when=lambda t, a: (_weeks(t) or 99) >= 20),
    Rule(r"(임신|산모).{0,30}(피|출혈)", code="Y0112", ceiling=3, when=lambda t, a: (_weeks(t) or 99) < 20),
    Rule(r"(임신|\d+\s*주).{0,25}(안\s*움직|태동)", code="Y0112", ceiling=3, negatable=False),
    Rule(r"아기\s*머리가?\s*보", code="Y0111", ceiling=2),
    Rule(r"진통|양수", code="Y0111", ceiling=3),
    Rule(r"진통|양수", code="Y0111", ceiling=2, when=lambda t, a: (_weeks(t) or 40) < 37),
    Rule(r"아랫배\s*한쪽.{0,12}(갑자기|찢어|심하)", code="Y0113", ceiling=2),
    Rule(r"생리.{0,10}늦.{0,30}(배|어지)", code="Y0113", ceiling=2),
    Rule(r"생리\s*양.{0,15}(많|패드)", code="Y0113", ceiling=3),

    # 외상·화상·절단
    Rule(r"끓는\s*물.{0,15}(뒤집어|온몸|가슴|배)|불이\s*붙|감전|(넓은|전신|온몸).{0,6}(화상|데)", code="Y0120", ceiling=2),
    Rule(r"목소리.{0,4}쉬|그을음", ceiling=1, when=lambda t, a: re.search(r"불|화상|연기", t)),
    Rule(r"(손가락|발가락).{0,15}(잘렸|잘린|절단|떨어져\s*나)", code="Y0131", ceiling=2),
    Rule(r"(손목|팔|다리|발목|발(?!가락)|손(?!가락|바닥|톱)).{0,15}(절단|잘렸|떨어져\s*나)", code="Y0132", ceiling=1),
    Rule(r"허벅지.{0,15}(꺾|튀어나|뼈)|뼈가?\s*.{0,6}튀어나", ceiling=2),
    Rule(r"목이?\s*아프.{0,15}(저려|저리)|(굴러|떨어|사고).{0,20}목", ceiling=2,
         when=lambda t, a: re.search(r"저려|저리|힘이", t)),
    Rule(r"핸들|가슴을\s*.{0,6}부딪", ceiling=2),

    # 출혈 (혈관중재)
    Rule(r"피가?\s*.{0,8}(멈추지|안\s*멈|계속)|피를\s*계속|(배|복부).{0,15}(부르고|부풀).{0,15}창백",
         code=_ped("Y0172", "Y0171"), ceiling=2, negatable=False),
    Rule(r"(넘어져|부딪|떨어져).{0,15}배.{0,20}창백", code="Y0172", ceiling=2, when=lambda t, a: child(a, 12)),

    # 투석
    Rule(r"투석", code="Y0141", ceiling=3),
    Rule(r"투석.{0,40}(숨|부었|맥박|기운|처지|의식|혈압)", code="Y0141", ceiling=2),
    Rule(r"투석.{0,40}(혈압(이)?\s*(계속\s*)?(낮|떨어)|의식)", code="Y0142", ceiling=2),
    Rule(r"투석.{0,40}(의식|혈압(이)?\s*[1-6]\d\s*(까지)?\s*(떨어|$))", code="Y0142", ceiling=1),

    # 정신과
    Rule(r"환청|헛것|칼을\s*들|위협|흥분해서|자해", code="Y0150", ceiling=2, when=lambda t, a: a is None or a >= 10),

    # 안과
    Rule(r"눈에.{0,10}(박혔|찔|튀었|들어갔)|(락스|세제|화학|약품).{0,10}눈|눈이.{0,10}(빨개지고\s*아프|안\s*보)",
         code="Y0160", ceiling=2, negatable=False),

    # 기타 긴급
    Rule(r"(입술|목|혀).{0,6}(붓|부어|부었)|(먹고|쏘이고|약을).{0,20}(두드러기|숨이)", ceiling=2),
    Rule(r"가스\s*냄새|일산화탄소|연탄", ceiling=2),
    Rule(r"과일\s*냄새", ceiling=2),
    Rule(r"혈압이?\s*(낮|떨어)|저혈압", ceiling=2),
]

# 급성이 아님을 나타내는 표현 (KTAS 1 규칙은 영향 없음)
NON_ACUTE = re.compile(
    r"예전에|과거에|원래|평소랑\s*비슷|평소에도|평소처럼|(지금은|이제는|요즘은)\s*괜찮"
    r"|(한|몇)\s*달|개월\s*(째|됐|전)|년\s*(째|됐|전)|약\s*받으러|처방"
)

# 부정 표현: 규칙에 걸린 구간 안에 있으면 그 규칙은 무시 (예: "가슴은 안 아프고")
NEGATION = re.compile(r"안\s|않|없")

# 증상 어휘: 하나도 없으면 무슨 일인지 모르는 것으로 보고 되묻는다
SYMPTOM = re.compile(
    r"머리|두통|가슴|흉통|배|복통|명치|옆구리|허리|등|목|팔|다리|손|발|무릎|발목|손목|어깨|엉덩|고관절|눈|귀|코|입|"
    r"얼굴|피부|치아|숨|호흡|기침|가래|열|오한|토|설사|변|소변|피|출혈|의식|쓰러|경련|발작|마비|힘이|저리|저려|"
    r"어지|두근|심장|맥|혈압|혈당|부었|붓|두드러기|가려|화상|데었|삼켰|삼킨|마셨|베였|찢어|잘렸|절단|뼈|삐|접질|"
    r"멍|곪|여드름|콧물|감기|임신|진통|양수|생리|환청|헛것|불안|잠을|자해|위협|약|술|떨어|부딪|박았|빠졌|사고|넘어|"
    r"찔렸|감전|딸꾹질|체해|속이|침을|사레|땅콩|동전|건전지|자석|세제|락스|낳|생후|투석|노래|노랗|횡설|정신"
)


def scan(text: str, age: Age) -> tuple[list[str], int | None]:
    """문장에서 규칙에 걸린 질환 코드 목록과 가장 엄격한 KTAS 상한을 돌려준다"""
    non_acute = bool(NON_ACUTE.search(text))
    codes: list[str] = []
    ceiling: int | None = None
    for rule in RULES:
        if non_acute and rule.acute_only:
            continue
        m = rule.regex.search(text)
        if not m:
            continue
        if rule.acute_only and rule.negatable and NEGATION.search(m.group(0)):
            continue
        if rule.when and not rule.when(text, age):
            continue
        code = rule.code(text, age) if callable(rule.code) else rule.code
        if code and code not in codes:
            codes.append(code)
        if rule.ceiling is not None:
            ceiling = rule.ceiling if ceiling is None else min(ceiling, rule.ceiling)
    return codes, ceiling


def has_symptom(text: str) -> bool:
    return bool(SYMPTOM.search(text))


def is_non_acute(text: str) -> bool:
    return bool(NON_ACUTE.search(text))


# ── 질환 코드 정합성 ───────────────────────────────────────────────────
CHILD_CODES = {"Y0070", "Y0082", "Y0092", "Y0100", "Y0172"}
ADULT_TWIN = {"Y0081": "Y0082", "Y0091": "Y0092", "Y0171": "Y0172"}   # 성인 코드 → 영유아 코드
OBSTETRIC = re.compile(r"임신|산모|양수|진통|출산|낳|분만|태동|\d+\s*주")
GYNECOLOGIC = re.compile(r"생리|아랫배|질\s*출혈|난소|자궁|임신")
HEAD_TRAUMA = re.compile(r"머리|두부|넘어|사고|부딪|떨어|박")


def plausible(code: str, text: str, age: Age) -> str | None:
    """나이·문맥과 맞지 않는 코드는 None, 성인/영유아 쌍은 나이에 맞게 바꿔서 돌려준다"""
    if age is not None:
        twin_child = ADULT_TWIN.get(code)
        twin_adult = {v: k for k, v in ADULT_TWIN.items()}.get(code)
        if twin_child and age <= 6:
            code = twin_child
        elif twin_adult and age > 6:
            code = twin_adult
        if code in CHILD_CODES and age > 6:
            return None
        if code in {"Y0051", "Y0052"} and age <= 6:     # 영유아 담낭·담도 질환은 드묾
            return None
        if code == "Y0020" and age < 15:
            return None
        if code == "Y0150" and age < 10:
            return None
    if code in {"Y0111", "Y0112"} and not OBSTETRIC.search(text):
        return None
    if code == "Y0113" and not GYNECOLOGIC.search(text):
        return None
    if code == "Y0032" and not HEAD_TRAUMA.search(text):
        return None
    return code


# 질환 코드만 있고 규칙 상한이 없을 때 쓰는 기본 상한 (LLM이 코드는 맞히고 KTAS만 낮게 주는 경우 대비)
CODE_CEILING = {
    "Y0010": 2, "Y0020": 2, "Y0031": 2, "Y0032": 2, "Y0041": 2, "Y0042": 2,
    "Y0051": 3, "Y0052": 3, "Y0060": 3, "Y0070": 2, "Y0081": 3, "Y0082": 3,
    "Y0091": 3, "Y0092": 3, "Y0100": 2, "Y0111": 3, "Y0112": 2, "Y0113": 3,
    "Y0120": 2, "Y0131": 2, "Y0132": 2, "Y0141": 3, "Y0142": 2, "Y0150": 2,
    "Y0160": 2, "Y0171": 2, "Y0172": 2,
}
