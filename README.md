# TradeGo — 수입단가 분석 플랫폼

관세청 무역통계 Open API 데이터를 기반으로 품목별 수입단가 추이를 조회하고, 가격 예측 모델을 검증해 가장 신뢰할 수 있는 방식으로 분석 결과를 제공하는 Flask 웹 서비스입니다.

## 모델 검증 과정

가격 예측에 머신러닝/시계열 모델을 적용했으나, Walk-forward Validation으로 엄격히 검증한 결과 단순 베이스라인보다 성능이 낮다는 것을 확인하고 **AI 예측 기능을 과감히 제외**했습니다. 성과를 부풀리기보다, 검증되지 않은 모델을 서비스에 노출시키지 않는 판단을 우선한 결과입니다.

| 모델 | MAPE | 비고 |
| --- | --- | --- |
| Prophet | 203.1% | 시계열 분해 기반, 변동성 큰 무역 데이터에 부적합 |
| XGBoost (8 features) | 84.8% | 피처를 늘려도 과적합 경향 |
| XGBoost (3 features) | 104.7% | 피처 축소 후에도 베이스라인 대비 열세 |
| **단순 베이스라인(직전 값 유지)** | **72.9%** | **최종 채택 — 가장 낮은 오차** |

## 주요 기능

- 관세청 무역통계 API 연동으로 품목(HS 코드)별 국가별 수입단가 실시간 조회
- Flask-Login 기반 회원가입/로그인, 분석 이력(AnalysisHistory) DB 저장 및 재조회
- Plotly 기반 가격 추이 시각화
- 동시 사용자 분석 요청 시 결과가 덮어써지던 전역 변수 구조를 사용자별 dict로 수정해 동시성 버그 해결

## 기술 스택

Python · Flask · Flask-Login · Flask-SQLAlchemy · PostgreSQL · Pandas · scikit-learn · XGBoost · Prophet · Plotly

## 실행 방법

```bash
pip install -r requirements.txt

# .env.example을 참고해 .env 파일을 만들고 본인 API 키 / DB 정보를 입력하세요
cp .env.example .env

python app.py
```

## 폴더 구조

```
app.py            # Flask 엔트리포인트, 라우팅
auth.py           # 회원가입/로그인
service.py        # 분석 로직 (API 호출 → 가공 → 결과 생성)
data_loader.py    # 관세청 API 연동
models.py         # DB 모델 (User, AnalysisHistory)
ml/               # 모델 학습/검증 스크립트 (features, train, model)
templates/        # 화면 템플릿
data/             # 환율·HS코드 참고 데이터
```
