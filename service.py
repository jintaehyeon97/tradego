import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from data_loader import fetch_trade_data_by_month_range

# =========================
# 🩹 이 파일은 대대적으로 단순화되었습니다.
#
# 배경: Prophet 예측(및 그 위에 XGBoost로 보정한 모델) 모두 실측 데이터로
# 검증한 결과, "6개월 뒤 가격 = 최근 평균가 그대로"라고 가정하는 가장 단순한
# naive 예측보다 정확도가 떨어졌습니다 (Prophet MAPE 203%, XGBoost 보정
# 84.8% vs naive 72.9%). 즉 이 데이터(국가별 월별 수입단가)는 6개월 뒤를
# 예측하기엔 노이즈가 너무 커서, 복잡한 모델을 얹을수록 오히려 나빠집니다.
#
# 그래서:
#   - Prophet 예측 제거 → '미래' 추정치는 avg를 그대로 사용 (검증된 최선)
#   - XGBoost 모델 제거 → AI점수는 투명한 규칙 기반 공식으로 대체
#   - 그래프용 데이터는 예측선이 아니라 실제 관측치(history)를 사용
#     (Prophet이 만들어내던 "스파이크" 문제가 구조적으로 사라짐)
#   - '트렌드'는 미래를 점치는 게 아니라, 과거 데이터 안에서 최근 흐름을
#     서술하는 값으로 재정의 (최근 3개월 vs 그 이전 3개월 비교)
# =========================

# =========================
# 국가 설정
# =========================
DEFAULT_MAJOR_COUNTRIES = [
    'CN','US','JP','DE','VN','TH','ID','IN','IT','MY',
    'FR','NL','CA','AU','MX','CZ','PL','TR','ES','GB'
]

COUNTRY_NAME_MAP = {
    'CN':'중국','US':'미국','JP':'일본','DE':'독일','VN':'베트남','TH':'태국',
    'ID':'인도네시아','IN':'인도','IT':'이탈리아','MY':'말레이시아','FR':'프랑스',
    'NL':'네덜란드','CA':'캐나다','AU':'호주','MX':'멕시코','CZ':'체코',
    'PL':'폴란드','TR':'터키','ES':'스페인','GB':'영국'
}


# =========================
# 리스크 (기존과 동일: 변동계수 기반)
# =========================
def get_risk(df):
    avg = df['unitPrice'].mean()
    cv = df['unitPrice'].std() / avg if avg else 0

    if cv > 0.4:
        return "높음", cv
    elif cv > 0.2:
        return "중간", cv
    return "낮음", cv


# =========================
# 🩹 FIX: 미래 예측 대신 "최근 흐름" 서술.
# 과거 데이터 안에서 최근 3개월 평균과 그 이전 3개월 평균을 비교합니다.
# 어디까지나 "최근에 이랬다"는 관찰이지, "앞으로 이럴 것이다"라는
# 예측이 아님을 명확히 하기 위해 함수/변수명도 forecast가 아닌
# recent_trend로 둡니다.
# =========================
def get_recent_trend(df):
    df = df.sort_values('yyyymm')

    if len(df) < 6:
        return "안정", 0.0

    recent = df['unitPrice'].iloc[-3:].mean()
    prior = df['unitPrice'].iloc[-6:-3].mean()

    if prior == 0:
        return "안정", 0.0

    change = (recent - prior) / prior
    change = float(np.clip(change, -0.5, 0.5))

    if change > 0.05:
        return "상승", change
    elif change < -0.05:
        return "하락", change
    return "안정", change


# =========================
# AI 추천 이유
# 🩹 '향후 가격 상승 가능성' 같은 예측성 문구를 '최근 ~한 흐름이었습니다'로
# 수정해서, 더 이상 하지 않는 예측을 하는 것처럼 보이지 않도록 함.
# =========================
def generate_ai_reason(row, avg_price, min_price):

    if row['평균_raw'] == min_price:
        price = "가격 경쟁력이 매우 높습니다"
    elif row['평균_raw'] < avg_price:
        price = "가격 경쟁력이 있는 수준입니다"
    else:
        price = "가격은 평균적인 수준입니다"

    if row['트렌드'] == "상승":
        trend = "최근 가격이 상승하는 흐름이었습니다"
    elif row['트렌드'] == "하락":
        trend = "최근 가격이 하락하는 흐름이었습니다"
    else:
        trend = "최근 가격은 안정적인 흐름이었습니다"

    if row['리스크'] == "높음":
        risk = "변동성이 있어 주의가 필요합니다"
    elif row['리스크'] == "중간":
        risk = "일정 수준의 변동성이 존재합니다"
    else:
        risk = "안정적인 거래가 가능합니다"

    return f"{price}. {trend}. {risk}."


# =========================
# 🩹 FIX: 동시에 처리할 국가 수.
# API가 동시 연결을 얼마나 잘 견디는지에 따라 조정하세요.
# 너무 크게 잡으면 오히려 API가 차단/거부하기 시작할 수 있어서,
# 5~8 정도로 시작해서 실패율 보면서 조정하는 걸 권장합니다.
# =========================
MAX_WORKERS = 5


def _fetch_one_country(country, start, end, hs_code):
    """국가 1곳의 데이터를 가져와 분석 결과 dict로 반환 (없으면 None)."""
    country_name = COUNTRY_NAME_MAP.get(country, country)

    df = fetch_trade_data_by_month_range(start, end, hs_code, country)

    if df.empty:
        print(f"[{country_name}] 데이터 없음")
        return None

    # =========================
    # 🩹 FIX: 데이터가 너무 적으면(2~3개월) std()는 계산되지만
    # 통계적으로 사실상 의미가 없음 — 점 2개를 이은 직선이 "안정적인
    # 추세"로, 우연히 낮은 변동성이 "낮은 리스크"로 둔갑해서 오히려
    # 근거 없이 1등으로 추천되는 역효과가 생길 수 있음. 최소 6개월
    # 이상 있는 국가만 순위에 포함 (get_recent_trend의 기준과도 맞춤).
    # =========================
    MIN_MONTHS = 6
    if len(df) < MIN_MONTHS:
        print(f"[{country_name}] 데이터 부족({len(df)}개월, 최소 {MIN_MONTHS}개월 필요) → 제외")
        return None

    avg = df['unitPrice'].mean()
    risk, cv = get_risk(df)
    trend, change = get_recent_trend(df)

    # 🩹 '미래' 추정치 = avg. 실측 검증 결과 이게 Prophet/XGBoost보다 정확했음
    future = avg

    # 🩹 AI점수 = 투명한 규칙 기반 공식
    score = -avg * (1 + cv)

    return {
        '국가명': country_name,
        '평균_raw': avg,
        '미래_raw': future,
        '트렌드': trend,
        '리스크': risk,
        '변동률': change,
        'AI점수': score,
        'history': df[['yyyymm', 'unitPrice']].sort_values('yyyymm').to_dict('records'),
    }


# =========================
# 🔥 전체 분석
# 🩹 FIX: 국가별 순차 처리(for) → 병렬 처리(ThreadPoolExecutor)로 전환.
# 국가끼리는 서로 결과가 독립적이라 병렬로 돌려도 안전함.
# API 호출(네트워크 대기)이 대부분의 시간을 차지하므로, 동시에 여러 국가를
# 요청하면 전체 소요 시간이 MAX_WORKERS배 가까이 줄어듦.
# =========================
def run_full_analysis(start, end, hs_code, progress_callback=None):

    total = len(DEFAULT_MAJOR_COUNTRIES)
    results = []

    # 여러 스레드가 동시에 progress_callback을 부르니, 카운터 증가는
    # lock으로 보호해서 진행률이 꼬이지 않게 함
    completed = 0
    lock = threading.Lock()

    def run_and_report(country):
        nonlocal completed
        result = _fetch_one_country(country, start, end, hs_code)

        with lock:
            completed += 1
            if progress_callback:
                progress_callback({
                    "current": COUNTRY_NAME_MAP.get(country, country),
                    "step": completed,
                    "total": total,
                    "percent": int(completed / total * 100)
                })

        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_and_report, c) for c in DEFAULT_MAJOR_COUNTRIES]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    # 🔥 최소 결과 보장
    if not results:
        return pd.DataFrame([{
            '국가명': '데이터 부족',
            '평균_raw': 0,
            '미래_raw': 0,
            '트렌드': '-',
            '리스크': '-',
            '변동률': 0,
            'AI점수': 0,
            'history': []
        }])

    df = pd.DataFrame(results)

    # =========================
    # 🔥 AI 점수 정규화 (기존과 동일)
    # =========================
    min_score = df['AI점수'].min()
    max_score = df['AI점수'].max()

    if max_score - min_score == 0:
        df['AI점수'] = 50
    else:
        df['AI점수'] = (df['AI점수'] - min_score) / (max_score - min_score) * 100

    df['AI점수'] = df['AI점수'].round(1)

    df = df.sort_values('AI점수', ascending=False)

    avg_price = df['평균_raw'].mean()
    min_price = df['평균_raw'].min()

    df['AI추천이유'] = df.apply(
        lambda row: generate_ai_reason(row, avg_price, min_price),
        axis=1
    )

    if progress_callback:
        progress_callback({
            "current": "완료",
            "step": total,
            "total": total,
            "percent": 100
        })

    return df