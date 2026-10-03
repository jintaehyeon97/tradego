"""
학습 데이터 생성 스크립트 (Walk-forward)
=========================================
과거의 여러 시점(cutoff)마다:
  1) cutoff 시점까지의 데이터로만 피처 계산 (avg, Prophet 예측치, change, volatility, count)
  2) cutoff + HORIZON_MONTHS 시점에 실제로 관측된 가격을 라벨로 기록

이렇게 만들어야 모델이 "실제 미래 결과"를 학습합니다.
(기존 make_training_data.py는 라벨(label)이 avg/future/change의 선형 공식으로
 만들어져 있어서 모델이 배울 게 없었습니다 — 이번엔 label을 진짜 실측 가격으로 씁니다.
 컬럼명은 기존 train.py가 바로 읽을 수 있도록 avg/future/change/volatility/count/label로 유지.)

⚠️ 반드시 공공데이터포털 API 키가 유효한 로컬 환경에서 실행하세요.
⚠️ API 호출량이 매우 많습니다 (HS코드 수 × 국가 수 × cutoff 수 × 개월 수).
   HS_CODES와 cutoff 간격을 먼저 작게 잡고 테스트해본 뒤 늘리는 것을 권장합니다.

실행: 프로젝트 루트에서 `python ml/make_training_data.py` 로 실행하세요.
      (기존 스크립트와 동일하게 ml/ 폴더 안에 두는 걸 가정합니다)
"""

# =========================
# 🔥 경로 문제 해결 (기존 스크립트와 동일)
# ml/ 폴더 안에서 실행돼도 프로젝트 루트의 service.py, data_loader.py를
# import할 수 있도록 루트를 sys.path에 추가.
# =========================
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from prophet import Prophet

from data_loader import fetch_trade_data_by_month_range
from service import DEFAULT_MAJOR_COUNTRIES

# =========================
# 설정
# =========================
HORIZON_MONTHS = 6      # future가 몇 개월 뒤를 예측하는지 (기존 서비스와 동일하게 맞춤)
LOOKBACK_YEARS = 3      # 피처 계산에 사용할 과거 데이터 기간 (기존 run_prophet과 동일)
MIN_MONTHS = 8          # 최소 관측치 (기존 run_prophet 기준과 동일)
CUTOFF_STEP_MONTHS = 6  # cutoff 시점 간격

# 학습에 사용할 HS코드 목록 (50개).
# 🩹 앞 7개는 기존에 이미 다루던 품목(커피/코코넛류/솔트 등), 나머지 43개는
# cleaned_hs_code_lookup.csv의 87개 HS코드 챕터(카테고리) 중 겹치지 않는
# 챕터에서 하나씩 뽑아 다양성을 확보한 것 (곡물·화학·섬유·목재·수산물 등 혼합).
# 특정 카테고리/가격대에만 모델이 과적합되지 않도록 하기 위함.
HS_CODES = [
    # 기존 7개
    "1006301000", "1006302000", "2101111000", "2101121000",
    "2203000000", "2603000000", "2606000000",
    # 챕터 다양화 43개
    "1103191000", "0012099110", "1302311000", "1404904000",
    "1512112000", "1604141011", "1701121000", "1801001000",
    "1905400000", "2009810000", "2309102000", "2403993000",
    "2509000000", "2710196000", "2836200000", "0029362910",
    "0302210000", "3102400000", "3202901000", "3302101000",
    "3407002000", "3504002020", "3605000000", "3701100000",
    "3825902000", "3915903000", "4016991010", "4113100000",
    "4205001900", "4301100000", "4421913000", "4502000000",
    "4601211000", "4706200000", "4810992000", "4901101000",
    "0000507901", "5109109000", "5205470000", "5302902020",
    "5402200000", "5505100000", "5603920000",
]


def generate_cutoffs(start="201901", step_months=CUTOFF_STEP_MONTHS, horizon_months=HORIZON_MONTHS):
    start_date = datetime.strptime(start, "%Y%m")
    end_date = datetime.now() - relativedelta(months=horizon_months + 1)
    cutoffs = []
    d = start_date
    while d <= end_date:
        cutoffs.append(d.strftime("%Y%m"))
        d += relativedelta(months=step_months)
    return cutoffs


def run_prophet_asof(hist_df):
    """
    기존 service.run_prophet과 동일한 하이퍼파라미터를 쓰되,
    이 스크립트에서 필요한 forecast 전체를 반환.
    (train/serve skew 방지를 위해 실제 service.py와 설정을 반드시 맞출 것)
    """
    if len(hist_df) < MIN_MONTHS:
        return None

    df = hist_df.copy()
    df['ds'] = pd.to_datetime(df['yyyymm'], format='%Y%m')
    df['y'] = np.log(df['unitPrice'].clip(lower=0.1))

    model_p = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='additive',
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model_p.fit(df[['ds', 'y']])

    future = model_p.make_future_dataframe(periods=HORIZON_MONTHS, freq='M')
    forecast = model_p.predict(future)
    forecast['yhat'] = np.exp(forecast['yhat'])

    # =========================
    # 🩹 FIX: inf/극단값 원천 차단.
    # 로그 공간에서 트렌드가 튀면 exp() 복원 시 부동소수점 오버플로로
    # 실제 inf가 나올 수 있음(짧은 데이터일수록 잘 생김). service.py의
    # run_prophet에 넣었던 것과 동일하게, 실측 중앙값 기준 0.5~2배
    # 범위로 클리핑.
    # =========================
    hist_median = hist_df['unitPrice'].median()
    forecast['yhat'] = forecast['yhat'].clip(lower=hist_median * 0.5, upper=hist_median * 2.0)

    return forecast


# =========================
# 🩹 FIX: 캐싱 레이어
# cutoff마다 3년치를 매번 새로 API 호출하면 인접 cutoff끼리 겹치는
# 구간을 계속 중복 호출하게 됨. (country, hs_code) 당 필요한 전체
# 기간을 딱 한 번만 받아서 메모리에 캐싱하고, cutoff별로는 그 안에서
# 슬라이싱만 함 → API 호출량이 cutoff 개수만큼 나눠지는 대신
# "필요한 전체 개월 수"만큼으로 줄어듦 (보통 4배 이상 절감).
# =========================
_fetch_cache = {}


def fetch_full_range_cached(hs_code, country, range_start, range_end):
    key = (hs_code, country)
    if key not in _fetch_cache:
        print(f"  [캐시 미스] {hs_code} {country} {range_start}~{range_end} 전체 fetch")
        _fetch_cache[key] = fetch_trade_data_by_month_range(range_start, range_end, hs_code, country)
    return _fetch_cache[key]


def build_row(hs_code, country, cutoff, full_df):
    """
    cutoff 시점을 기준으로 피처를 계산하고,
    cutoff + HORIZON_MONTHS 시점의 실제 관측 가격을 라벨로 반환.
    full_df: 해당 (hs_code, country)의 전체 기간 데이터 (캐싱된 것)
    """
    cutoff_date = datetime.strptime(cutoff, "%Y%m")
    lookback_start = (cutoff_date - relativedelta(years=LOOKBACK_YEARS)).strftime("%Y%m")
    label_month = (cutoff_date + relativedelta(months=HORIZON_MONTHS)).strftime("%Y%m")

    # ① cutoff까지의 데이터만 슬라이싱 (미래 정보 절대 사용 금지)
    hist_df = full_df[(full_df['yyyymm'] >= lookback_start) & (full_df['yyyymm'] <= cutoff)]
    if hist_df.empty or len(hist_df) < MIN_MONTHS:
        return None

    avg = hist_df['unitPrice'].mean()
    cv = hist_df['unitPrice'].std() / avg if avg else 0

    forecast = run_prophet_asof(hist_df)
    if forecast is None:
        return None

    prophet_future = forecast.iloc[-1]['yhat']            # cutoff+HORIZON_MONTHS 시점 예측치
    recent = forecast.iloc[-(HORIZON_MONTHS + 1)]['yhat']
    # 🩹 service.py의 get_trend()와 동일한 클립 범위(±0.5)로 맞춤 (train/serve skew 방지)
    change = float(np.clip((prophet_future - recent) / recent, -0.5, 0.5))

    # =========================
    # 🩹 FIX: 0 나눗셈/극단값 방어
    # training_data_test.csv에서 avg가 0에 가까울 때 change/future_ratio가
    # 851까지 폭발하는 버그가 있었음. run_prophet에서 이미
    # df = df[df['unitPrice'] > 0.1] 로 걸러지긴 하지만, 혹시 모를 극단값에
    # 대한 최종 방어선으로 avg 자체가 비정상적으로 작으면 이 행을 버림.
    # =========================
    if avg < 0.1:
        return None

    # ② 라벨: cutoff + HORIZON_MONTHS 시점의 "실제" 관측 가격 (역시 캐싱된 full_df에서 슬라이싱)
    actual_row = full_df[full_df['yyyymm'] == label_month]
    if actual_row.empty:
        return None
    actual_price = actual_row['unitPrice'].mean()

    if actual_price <= 0 or actual_price / avg > 20 or actual_price / avg < 0.05:
        # 20배 폭등/95% 폭락처럼 사실상 다른 품목 오분류나 데이터 오류로
        # 보이는 극단적 케이스는 학습에서 제외 (진짜 급변 케이스라기엔
        # 너무 크므로, 이 임계값은 필요시 조정하세요)
        return None

    return {
        "hs_code": hs_code,
        "country": country,
        "cutoff": cutoff,
        "avg": avg,
        "future": prophet_future,   # 🩹 train.py와 컬럼명 맞춤 (기존: prophet_future)
        "change": change,
        "volatility": cv,
        "count": len(hist_df),
        "label": actual_price,      # 🩹 train.py와 컬럼명 맞춤 (기존: label_actual_price)
    }


def main():
    cutoffs = generate_cutoffs()

    # 캐싱을 위해 필요한 전체 기간 범위 계산
    earliest_cutoff = datetime.strptime(cutoffs[0], "%Y%m")
    latest_cutoff = datetime.strptime(cutoffs[-1], "%Y%m")
    range_start = (earliest_cutoff - relativedelta(years=LOOKBACK_YEARS)).strftime("%Y%m")
    range_end = (latest_cutoff + relativedelta(months=HORIZON_MONTHS)).strftime("%Y%m")

    total_combos = len(HS_CODES) * len(DEFAULT_MAJOR_COUNTRIES)
    done_combos = 0

    os.makedirs("ml", exist_ok=True)
    out_path = os.path.join("ml", "training_data_v2.csv")

    # =========================
    # 🩹 FIX: 재실행(resume) 지원.
    # 며칠씩 돌리는 작업이라 중간에 컴퓨터가 꺼지거나 재부팅되면
    # 처음부터 다시 돌리는 건 너무 아까움. 이미 결과 CSV에 기록된
    # (hs_code, country) 조합은 건너뛰고 이어서 진행.
    # =========================
    done_pairs = set()
    file_exists = os.path.exists(out_path)
    if file_exists:
        try:
            prev = pd.read_csv(out_path, dtype={"hs_code": str})
            done_pairs = set(zip(prev["hs_code"], prev["country"]))
            print(f"🩹 이어서 진행: 기존 {out_path}에서 {len(done_pairs)}개 조합 완료 확인 → 스킵")
        except Exception as e:
            print(f"⚠ 기존 파일 읽기 실패({e}) → 처음부터 새로 씀")
            file_exists = False

    print(f"HS코드 {len(HS_CODES)} x 국가 {len(DEFAULT_MAJOR_COUNTRIES)} x cutoff {len(cutoffs)}")
    print(f"조합당 전체 fetch 1회({range_start}~{range_end}) + cutoff별 라벨/피처는 슬라이싱만 수행")

    fieldnames = ["hs_code", "country", "cutoff", "avg", "future", "change", "volatility", "count", "label"]

    # 헤더는 파일이 아직 없을 때만 씀
    write_header = not file_exists
    total_new_rows = 0

    for hs_code in HS_CODES:
        for country in DEFAULT_MAJOR_COUNTRIES:
            done_combos += 1

            # 🩹 이미 끝난 조합은 API 호출 없이 바로 스킵
            if (hs_code, country) in done_pairs:
                print(f"[{done_combos}/{total_combos}] {hs_code} {country} → 이미 완료, 스킵")
                continue

            print(f"[{done_combos}/{total_combos}] {hs_code} {country}")

            combo_rows = []
            try:
                full_df = fetch_full_range_cached(hs_code, country, range_start, range_end)
            except Exception as e:
                print(f"  fetch 실패: {e}")
                full_df = pd.DataFrame()

            if not full_df.empty:
                for cutoff in cutoffs:
                    try:
                        row = build_row(hs_code, country, cutoff, full_df)
                        if row:
                            combo_rows.append(row)
                    except Exception as e:
                        print(f"  {cutoff} 실패: {e}")

            # =========================
            # 🩹 FIX: 조합 하나가 끝날 때마다 즉시 CSV에 append.
            # combo_rows가 0개(데이터 없어서 스킵된 조합)여도 "완료 표시"를
            # 위해 빈 마커 행이 아니라, 그냥 다음 조합으로 넘어감
            # (done_pairs 재구성 시 이 조합은 다시 시도될 수 있으나,
            #  데이터가 없는 조합은 재시도해도 어차피 다시 없음 → 손해 적음)
            # =========================
            if combo_rows:
                combo_df = pd.DataFrame(combo_rows, columns=fieldnames)
                combo_df.to_csv(out_path, mode="a", index=False, header=write_header)
                write_header = False
                total_new_rows += len(combo_rows)
                print(f"  → {len(combo_rows)}행 저장 (누적 신규 {total_new_rows}행)")

    print(f"\n완료. 결과 파일: {out_path}")


if __name__ == "__main__":
    main()
