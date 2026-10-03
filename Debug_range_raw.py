"""
range 호출 원문 응답 확인
==========================
items가 0건으로 나온 이유를 알아내기 위해, API가 실제로 뭘 돌려주는지
그대로 출력합니다. 에러 메시지일 수도 있고, 정상 응답인데 구조가
예상과 다를 수도 있습니다.
"""

import requests
from config import API_KEY

URL = 'http://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList'

TEST_HS_CODE = "1006301000"
TEST_COUNTRY = "CN"
TEST_START = "202301"
TEST_END = "202606"

params = {
    'serviceKey': API_KEY,
    'strtYymm': TEST_START,
    'endYymm': TEST_END,
    'hsSgn': TEST_HS_CODE,
    'cntyCd': TEST_COUNTRY,
    'numOfRows': 9999,
    'pageNo': 1,
}

res = requests.get(URL, params=params, timeout=20)

print("=== HTTP 상태 코드 ===")
print(res.status_code)
print("\n=== 응답 원문 (앞 3000자) ===")
print(res.text[:3000])

print("\n\n=== 비교: 기존 방식(단일 월)으로 같은 국가/HS코드 호출 시 응답 원문 ===")
params_single = dict(params)
params_single['strtYymm'] = TEST_START
params_single['endYymm'] = TEST_START  # 시작=끝 (기존 방식과 동일)
res_single = requests.get(URL, params=params_single, timeout=20)
print(res_single.text[:3000])