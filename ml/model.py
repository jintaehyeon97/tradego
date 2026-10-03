import joblib
import numpy as np

# =========================
# 🩹 FIX: train.py는 "ml/xgb_model.pkl"로 저장하는데
# 여기는 "ml/saved_model.pkl"을 로드하고 있었음 (파일이 존재하지 않아
# 이 모듈을 실제로 쓰면 무조건 에러가 나는 상태였음).
# service.py가 쓰는 경로(ml/xgb_model.pkl)와 통일.
# =========================
MODEL_PATH = "ml/xgb_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_score(feature_dict):
    """
    ⚠️ feature_dict는 반드시 features.FEATURE_ORDER와 같은 순서/키로
    구성되어 있어야 합니다. dict라도 파이썬 3.7+는 삽입 순서를 보장하지만,
    실수로 순서가 어긋나면 XGBoost가 조용히 잘못된 값을 예측하니
    가능하면 아래처럼 FEATURE_ORDER를 기준으로 명시적으로 순서를 맞추세요.
    """
    from features import FEATURE_ORDER
    X = np.array([[feature_dict[k] for k in FEATURE_ORDER]])
    return model.predict(X)[0]
