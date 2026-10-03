# =========================
# 🔥 라이브러리
# =========================
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

# 🩹 FIX: 피처 계산식을 features.py와 공유 (학습-서빙 불일치 방지용 단일 기준)
from features import add_batch_features, FEATURE_ORDER

# =========================
# 🔥 데이터 로드
# =========================
df = pd.read_csv("ml/training_data_v2.csv")

print("📊 원본 데이터 개수:", len(df))

# =========================
# 🩹 FIX: label 기준 이상치 제거를 '분위수'가 아니라 '방어적 필터'로.
# label이 이제 실제 가격이라, 분위수로 자르면 정상적인 고가/저가
# 품목까지 통째로 잘릴 수 있음. make_training_data.py에서 이미
# avg 대비 20배~0.05배 벗어나는 극단값은 걸러졌으므로 여기서는
# 남은 결측/음수만 방어적으로 제거.
# =========================
df = df[(df['label'] > 0) & (df['avg'] > 0) & (df['future'] > 0)]
print("📊 이상치 제거 후 데이터:", len(df))

# =========================
# 🩹 FIX: inf/NaN 방어.
# Prophet이 짧은 데이터로 극단적인 트렌드를 잡으면 log→exp 복원 과정에서
# 부동소수점 오버플로로 실제 inf가 나오는 경우가 있음. XGBoost는 inf가
# 하나라도 섞여있으면 전체 학습이 통째로 실패하므로 여기서 제거.
# =========================
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['avg', 'future', 'change', 'volatility', 'count', 'label'])
removed = before - len(df)
if removed:
    print(f"⚠ inf/NaN 값 포함된 {removed}행 제거")

# =========================
# 🩹 FIX: inf는 아니지만 터무니없이 큰 유한값 방어.
# XGBoost는 내부적으로 float32로 변환하는데, 값이 지나치게 크면
# (예: Prophet이 과거에 튀었을 때 남은 1e300 수준의 값) 그 변환
# 과정에서 사실상 inf가 되어 위와 똑같은 에러가 남. np.isinf()로는
# 못 잡으므로 크기 자체로 한 번 더 걸러냄.
# 단가(avg/future)가 이 정도로 클 리 없다는 상식적 상한선.
# =========================
MAX_REASONABLE = 1_000_000  # 필요시 데이터 특성에 맞게 조정
before2 = len(df)
df = df[(df['avg'].abs() < MAX_REASONABLE) & (df['future'].abs() < MAX_REASONABLE)]
removed2 = before2 - len(df)
if removed2:
    print(f"⚠ 비정상적으로 큰 값(avg/future) {removed2}행 제거")

print("📊 정제 후 데이터:", len(df))

# =========================
# 🔥 Feature Engineering
# 🩹 FIX: 여기서 직접 계산하지 않고 features.py의 add_batch_features를
# 그대로 호출. train.py와 service.py가 서로 다른 공식을 쓰다가
# 어긋나는 걸 막기 위함 (기존엔 여기 로직이 features.py랑 따로 놀았음).
# =========================
df = add_batch_features(df)

# 🩹 피처 엔지니어링 과정에서 새로 생긴 inf/극단값도 한 번 더 방어
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_ORDER + ['label'])
extreme_mask = (df[FEATURE_ORDER].abs() < MAX_REASONABLE).all(axis=1)
if (~extreme_mask).sum():
    print(f"⚠ 피처 엔지니어링 후 극단값 {(~extreme_mask).sum()}행 추가 제거")
df = df[extreme_mask]
print("📊 피처 엔지니어링 후 최종 데이터:", len(df))

# =========================
# 🩹 FIX: label 0~100 정규화 삭제.
# 예전엔 label이 사람이 만든 "점수"라 0~100으로 눌러도 의미가 보존됐지만,
# 지금 label은 "실제 관측 가격"(통화 단위) 그 자체를 맞히는 게 목표라서
# 정규화하면 오히려 무엇을 예측하는 모델인지 불분명해짐.
# =========================

# =========================
# 🔥 Feature / Target
# =========================
X = df[FEATURE_ORDER]
y = df['label']

# =========================
# 🩹 FIX: 랜덤 분리 → 시간 기준 분리.
# train_test_split(shuffle 기본 True)로 나누면 같은 국가/HS코드의
# 인접 cutoff가 train/test에 섞여 들어가 실제보다 훨씬 좋아 보이는
# 착시(leakage)가 생김. cutoff 기준으로 과거/최근을 나눔.
# =========================
df = df.sort_values('cutoff').reset_index(drop=True)
X = df[FEATURE_ORDER]
y = df['label']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_df = df.iloc[split_idx:]

print(f"학습: {len(X_train)}행 (~{df['cutoff'].iloc[:split_idx].min()} ~ {df['cutoff'].iloc[:split_idx].max()})")
print(f"검증: {len(X_test)}행  (~{test_df['cutoff'].min()} ~ {test_df['cutoff'].max()})")

# =========================
# 🔥 기본 모델 + 하이퍼파라미터 튜닝
# 🩹 GridSearchCV의 cv도 랜덤 섞임(KFold) 대신 시간 순서를 지키는
# TimeSeriesSplit으로 교체 (마찬가지로 leakage 방지)
# =========================
model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],        # 데이터가 아직 크지 않으므로 너무 깊게는 X
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='neg_mean_absolute_percentage_error',
    verbose=1,
    n_jobs=-1
)

print("\n🚀 모델 학습 시작...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\n✅ Best Params:")
print(grid.best_params_)

# =========================
# 🔥 평가
# =========================
y_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

# =========================
# 🩹 FIX: baseline(Prophet 원본 예측을 보정 없이 그대로 쓴 경우)과 비교.
# 이게 없으면 "모델이 좋아 보이는데 사실 Prophet 원본보다 못할 수도"를
# 놓치게 됨.
# =========================
baseline_pred = test_df['future'].values
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mape = mean_absolute_percentage_error(y_test, baseline_pred) * 100

# =========================
# 🩹 FIX: naive baseline 추가.
# feature importance에서 'future'(Prophet 예측)의 기여도가 매우 낮게
# 나올 경우, "그냥 최근 평균가가 미래에도 유지된다"고 가정하는 가장
# 단순한 예측이 Prophet+XGBoost 조합과 얼마나 차이나는지 반드시 확인.
# 여기서 지면 Prophet을 쓸 이유 자체가 없어짐.
# =========================
naive_pred = test_df['avg'].values
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_mape = mean_absolute_percentage_error(y_test, naive_pred) * 100

print("\n📊 모델 성능 (검증셋, 시간 기준 분리)")
print(f"{'':14}{'MAE':>10}{'MAPE':>10}{'RMSE':>10}{'R2':>8}")
print(f"{'모델(보정)':14}{mae:>10.4f}{mape:>9.2f}%{rmse:>10.4f}{r2:>8.3f}")
print(f"{'Prophet 원본':14}{baseline_mae:>10.4f}{baseline_mape:>9.2f}%")
print(f"{'naive(avg 유지)':14}{naive_mae:>10.4f}{naive_mape:>9.2f}%")

improvement = (baseline_mape - mape) / baseline_mape * 100
naive_improvement = (naive_mape - mape) / naive_mape * 100
print(f"\nProphet 대비 개선율: {improvement:+.1f}%")
print(f"naive 대비 개선율:   {naive_improvement:+.1f}%")

# =========================
# 🩹 실험: 노이즈 많은 피처(future/change/future_ratio/stability 등)를
# 빼고 중요도 상위였던 avg/price_rank 위주로만 축소한 모델도 같이 학습.
# feature importance 상 avg+price_rank가 64%를 차지했는데, 나머지
# 피처들이 오히려 노이즈로 작용해 naive보다 못한 결과가 나왔을 수도
# 있어서, 이걸 걷어내면 naive를 이길 수 있는지 확인.
# =========================
SIMPLE_FEATURES = ['avg', 'price_rank', 'count']
X_train_s = df.iloc[:split_idx][SIMPLE_FEATURES]
X_test_s = df.iloc[split_idx:][SIMPLE_FEATURES]

simple_model = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
)
simple_model.fit(X_train_s, y_train)
simple_pred = simple_model.predict(X_test_s)
simple_mape = mean_absolute_percentage_error(y_test, simple_pred) * 100
simple_mae = mean_absolute_error(y_test, simple_pred)

print(f"\n📊 축소 모델 실험 (avg, price_rank, count 만 사용)")
print(f"{'축소 모델':14}{simple_mae:>10.4f}{simple_mape:>9.2f}%")
print(f"naive 대비 개선율: {(naive_mape - simple_mape) / naive_mape * 100:+.1f}%")

# =========================
# 🔥 Feature 중요도
# =========================
print("\n📊 Feature 중요도")
importance = best_model.feature_importances_
for i, col in enumerate(FEATURE_ORDER):
    print(f"{col}: {importance[i]:.4f}")

# =========================
# 🔥 모델 저장
# 🩹 FIX: baseline보다 나을 때만 저장 + 기존 xgb_model.pkl을 바로
# 덮어쓰지 않고 별도 파일로 저장 (검증 후 수동 교체 권장)
# =========================
if mape < baseline_mape and mape < naive_mape:
    joblib.dump(best_model, "ml/xgb_model_v2.pkl")
    print("\n✅ 모델이 Prophet 원본과 naive 둘 다 이겼습니다 → ml/xgb_model_v2.pkl 저장 완료")
    print("   확인 후 교체: mv ml/xgb_model.pkl ml/xgb_model_backup.pkl && mv ml/xgb_model_v2.pkl ml/xgb_model.pkl")
elif simple_mape < naive_mape:
    joblib.dump(simple_model, "ml/xgb_model_simple_v2.pkl")
    print("\n✅ 8피처 모델은 naive를 못 이겼지만, 축소 모델(avg/price_rank/count)은 naive를 이겼습니다")
    print("   → ml/xgb_model_simple_v2.pkl 저장 완료 (단, 이 모델은 피처가 3개뿐이라")
    print("     service.py의 make_features/model.predict 호출 부분도 함께 3피처로 바꿔야 합니다)")
elif mape < baseline_mape:
    print("\n⚠️ 모델이 Prophet 원본보다는 낫지만, naive(그냥 최근 평균가 유지)보다는 못합니다.")
    print("   → 저장하지 않았습니다. 지금 구조(Prophet+XGBoost)가 가장 단순한 예측보다")
    print("     나은 게 없다는 뜻이라, 서비스에 반영하지 않는 게 맞습니다.")
else:
    print("\n⚠️ 모델이 Prophet 원본보다 나아지지 않아 저장하지 않았습니다.")
    print("   데이터를 더 모으거나 피처/하이퍼파라미터를 조정해보세요.")
