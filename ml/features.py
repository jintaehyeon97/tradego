"""
피처 생성 함수
================
🩹 FIX: 기존 features.py는 avg/future/change/volatility/count 5개만 만들었는데,
train.py는 price_rank/future_ratio/stability를 추가한 8개로 학습합니다.
이 불일치 때문에 재학습된 모델을 그대로 서비스에 연결하면
입력 차원이 안 맞아서 에러가 납니다 (model.predict가 8개를 기대하는데 5개만 줌).

여기서는 두 단계로 나눴습니다:
  1) make_base_features(df, forecast)  → 국가 1곳 단위로 계산 가능한 5개
  2) add_batch_features(batch_df)      → 같은 HS코드의 여러 국가를 한 번에
     모아놓은 데이터프레임에서만 계산 가능한 3개 (price_rank는 "같은 HS코드
     그룹 내 상대 순위"라서 국가 하나만 봐서는 계산이 안 되고, 여러 국가를
     모은 배치 단위로 계산해야 합니다)

⚠️ service.py에 연결할 때는 지금처럼 국가 루프 안에서 바로
   model.predict를 부르는 방식이 아니라, 루프가 다 끝나고
   전체 결과를 모은 뒤(add_batch_features) 한 번에 predict 해야 합니다.
   (아직 service.py는 이 구조로 안 바뀌어 있으니, 재학습된 모델을 실제로
   붙이실 때 같이 고쳐드릴게요)
"""

import numpy as np
import pandas as pd


def make_base_features(df, forecast):
    """
    국가 1곳의 데이터만으로 계산 가능한 기본 피처 5개.
    (기존 features.py의 make_features와 동일한 역할)
    """
    avg = df['unitPrice'].mean()
    future = forecast.iloc[-1]['yhat']
    change = (future - avg) / avg if avg else 0
    volatility = df['unitPrice'].std() / avg if avg else 0
    count = len(df)

    return {
        "avg": avg,
        "future": future,
        "change": change,
        "volatility": volatility,
        "count": count,
    }


def add_batch_features(batch_df):
    """
    같은 HS코드에 대해 여러 국가(예: 20개국)의 결과를 모은
    DataFrame(반드시 'avg', 'future', 'volatility' 컬럼 포함)에
    price_rank / future_ratio / stability 3개 컬럼을 추가해서 반환.

    train.py의 Feature Engineering 단계와 완전히 동일한 계산식이어야
    학습-서빙 불일치가 안 생깁니다. 로직 바뀌면 train.py도 같이 고치세요.
    """
    batch_df = batch_df.copy()

    # 🔹 품목(=이 배치) 내 상대 가격 순위
    batch_df['price_rank'] = batch_df['avg'].rank(pct=True)

    # 🔹 미래 대비 현재 비율
    batch_df['future_ratio'] = batch_df['future'] / (batch_df['avg'] + 1e-6)

    # 🔹 안정성 지표
    batch_df['stability'] = 1 / (batch_df['volatility'] + 0.01)

    return batch_df


# train.py, service.py 양쪽에서 동일한 순서를 보장하기 위한 단일 기준.
# (모델 학습 시 컬럼 순서와 추론 시 컬럼 순서가 다르면 XGBoost가
#  조용히 잘못된 값을 예측하니, 반드시 이 리스트 순서를 그대로 사용할 것)
FEATURE_ORDER = [
    "avg", "future", "change", "volatility", "count",
    "price_rank", "future_ratio", "stability",
]
