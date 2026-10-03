import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
from config import API_KEY

URL = 'https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList'

# 🩹 FIX: API 호출 timeout (초). 응답이 없으면 여기서 예외를 던지고
# 스레드가 무한 대기하지 않도록 함.
REQUEST_TIMEOUT = 10

# 🩹 FIX: 한 달치 요청이 실패했을 때 재시도 횟수
MAX_RETRIES = 2


def load_exchange_rate():
    df = pd.read_csv("data/USD_KRW_monthly.csv")
    df['yyyymm'] = df['yyyymm'].astype(str)
    return df


def fetch_trade_data(start, end, hs_code, country):

    params = {
        'serviceKey': API_KEY,
        'strtYymm': start,
        'endYymm': end,
        'hsSgn': hs_code,
        'cntyCd': country
    }

    # =========================
    # 🩹 FIX: timeout + 예외를 여기서 처리.
    # 실패하면 예외를 던지지 않고 "빈 DataFrame"을 반환해서
    # 호출부(월별 루프)가 그 달만 건너뛰고 계속 진행하게 함.
    # =========================
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.get(URL, params=params, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            break
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            print(f"[fetch_trade_data] {country} {start} 실패 (시도 {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                return pd.DataFrame()
            time.sleep(1)

    items = root.findall(".//item")

    data = []

    for item in items:

        # =========================
        # 총계 데이터 제외
        # hsCd가 "-"이면 총계
        # =========================
        if item.findtext("hsCd") == "-":
            continue

        data.append({
            'impDlr': float(item.findtext('impDlr') or 0),
            'impWgt': float(item.findtext('impWgt') or 0),
        })

    return pd.DataFrame(data)


def fetch_trade_data_by_month_range(start, end, hs_code, country):

    start_date = datetime.strptime(start, "%Y%m")
    end_date = datetime.strptime(end, "%Y%m")

    all_data = []

    while start_date <= end_date:

        yyyymm = start_date.strftime("%Y%m")

        # 🩹 FIX: 월 단위 호출 자체가 실패해도(위에서 빈 DataFrame 반환)
        # 예외가 여기까지 전파되지 않으므로 전체 루프는 계속 진행됨.
        df = fetch_trade_data(
            yyyymm,
            yyyymm,
            hs_code,
            country
        )

        if not df.empty:
            df['yyyymm'] = yyyymm
            all_data.append(df)

        start_date += relativedelta(months=1)
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data)

    # 중량이 0인 데이터 제거
    df = df[df['impWgt'] > 0]

    # 🩹 참고: 저거래량 왜곡 방지용 필터 (필요시 주석 해제해서 사용)
    # df = df[df['impWgt'] >= 100]

    # 중복 제거 (안전장치)
    df = df.drop_duplicates()

    df['qty'] = df['impWgt']
    df['unit'] = 'kg'
    df['unitPrice'] = df['impDlr'] / df['qty']

    return df[['yyyymm', 'unitPrice', 'qty', 'unit']]
