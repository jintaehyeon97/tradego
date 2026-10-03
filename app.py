from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from flask_login import LoginManager, login_required, current_user
import pandas as pd
import plotly.graph_objects as go
import plotly
import json
import threading

from service import run_full_analysis
from models import db, User, AnalysisHistory
from auth import auth_bp
import config

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "로그인이 필요합니다."
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


app.register_blueprint(auth_bp)

with app.app_context():
    db.create_all()  # 테이블이 없으면 자동 생성 (이미 있으면 아무 일도 안 함)


# =========================
# 🩹 FIX: 전역 변수 → 사용자별 dict.
# 기존엔 analysis_result/progress_status/last_request_params가 전역
# 변수 하나뿐이라, 동시에 여러 사용자가 쓰면 서로 결과가 덮어써지는
# 문제가 있었음. user_id를 key로 쓰는 dict로 바꿔서 사용자별로 분리.
#
# ⚠️ 참고: 여전히 메모리(dict)에만 있어서 서버 재시작하면 "진행 중이던"
# 상태는 날아갑니다. 완료된 분석은 DB(AnalysisHistory)에 저장되니 안전
# 하고, "진행 중" 상태만 휘발성이라 그 정도는 지금 규모에선 괜찮습니다.
# =========================
progress_status_by_user = {}
analysis_result_by_user = {}
last_request_params_by_user = {}


def get_progress(user_id):
    return progress_status_by_user.get(user_id, {
        "current": "", "step": 0, "total": 0, "percent": 0, "status": "idle"
    })


# =========================
# 데이터 로드 (기존과 동일)
# =========================
def load_hs_codes():
    df = pd.read_csv("data/cleaned_hs_code_lookup.csv", dtype={'HS코드': str})
    df['HS코드'] = df['HS코드'].str.zfill(10)
    return df[['품목명', 'HS코드']].drop_duplicates()


def load_exchange_rate():
    df = pd.read_csv("data/USD_KRW_monthly.csv")
    return df['exchange_rate'].iloc[-1]


def get_item_name(hs_code):
    df = load_hs_codes()
    row = df[df['HS코드'] == hs_code]
    if not row.empty:
        return row.iloc[0]['품목명']
    return hs_code


# =========================
# 백그라운드 분석
# =========================
def run_analysis_async(user_id, start, end, hs_code):
    progress_status_by_user[user_id] = {
        "current": "시작", "step": 0, "total": 0, "percent": 0, "status": "running"
    }

    def update_progress(status):
        progress_status_by_user.setdefault(user_id, {}).update(status)

    try:
        df = run_full_analysis(start, end, hs_code, progress_callback=update_progress)

        analysis_result_by_user[user_id] = df
        progress_status_by_user[user_id]["status"] = "done"

        # =========================
        # 🩹 분석 완료 시 DB에 기록 저장 (로그인 기능의 핵심)
        # =========================
        with app.app_context():
            item_name = get_item_name(hs_code)
            currency = last_request_params_by_user.get(user_id, {}).get("currency", "USD")

            history = AnalysisHistory(
                user_id=user_id,
                hs_code=hs_code,
                item_name=item_name,
                start=start,
                end=end,
                currency=currency,
                result_json=df.to_json(orient="records", force_ascii=False),
            )
            db.session.add(history)
            db.session.commit()

    except Exception as e:
        print(f"[run_analysis_async] 분석 실패: {e}")
        progress_status_by_user.setdefault(user_id, {})["status"] = "error"
        progress_status_by_user[user_id]["error_message"] = str(e)


# =========================
# 🩹 결과 렌더링 공통 함수.
# 실시간 분석 결과(result_internal)와 과거 기록 재생(history_detail)이
# 그래프/테이블을 만드는 로직을 공유하도록 분리.
# =========================
def render_analysis_df(df, hs_code, start, end, currency, item_name=None):
    if df is None or df.empty:
        return render_template(
            'index.html',
            tables=[],
            hs_list=load_hs_codes().to_dict('records'),
            selected_currency=currency
        )

    df = df.copy()
    exchange_rate = load_exchange_rate()

    if currency == "KRW":
        df['평균'] = df['평균_raw'] * exchange_rate
        unit = "원/kg"
    else:
        df['평균'] = df['평균_raw']
        unit = "USD/kg"

    df['평균'] = df['평균'].apply(lambda x: f"{int(x):,} {unit}" if currency == "KRW" else f"{x:.2f} {unit}")

    selected_item_name = item_name or get_item_name(hs_code)

    fig = go.Figure()
    colors = ['#2563eb', '#10b981', '#f59e0b']

    for i, row in enumerate(df.head(3).to_dict('records')):
        history = row.get('history', [])
        x = [pd.to_datetime(r['yyyymm'], format='%Y%m') for r in history]
        y = [r['unitPrice'] for r in history]

        if currency == "KRW":
            y = [v * exchange_rate for v in y]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', name=row['국가명'],
            line=dict(width=3, color=colors[i])
        ))

    fig.update_layout(template="plotly_dark")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        'index.html',
        tables=df.to_dict('records'),
        graphJSON=graphJSON,
        hs_list=load_hs_codes().to_dict('records'),
        top_country=df.iloc[0].to_dict(),
        selected_currency=currency,
        hs_code=hs_code,
        start=start,
        end=end,
        selected_item_name=selected_item_name
    )


def result_internal(user_id, hs_code, start, end, currency):
    df = analysis_result_by_user.get(user_id)
    return render_analysis_df(df, hs_code, start, end, currency)


# =========================
# 메인 페이지
# =========================
@app.route('/')
@login_required
def index():
    user_id = current_user.id
    last_params = last_request_params_by_user.get(user_id)
    analysis_result = analysis_result_by_user.get(user_id)

    if analysis_result is not None and last_params is not None:
        return result_internal(
            user_id,
            last_params['hs_code'],
            last_params['start'],
            last_params['end'],
            last_params['currency']
        )

    return render_template(
        'index.html',
        hs_list=load_hs_codes().to_dict('records')
    )


# =========================
# 분석 시작
# =========================
@app.route('/start_analysis', methods=['POST'])
@login_required
def start_analysis():
    user_id = current_user.id

    hs_code = request.form['hs_code']
    start = request.form['start']
    end = request.form['end']
    currency = request.form['currency']

    last_request_params_by_user[user_id] = {
        "hs_code": hs_code, "start": start, "end": end, "currency": currency
    }
    analysis_result_by_user[user_id] = None

    thread = threading.Thread(
        target=run_analysis_async,
        args=(user_id, start, end, hs_code)
    )
    thread.start()

    return jsonify({"status": "started"})


@app.route('/progress')
@login_required
def progress():
    return jsonify(get_progress(current_user.id))


@app.route('/result', methods=['POST'])
@login_required
def result():
    return result_internal(
        current_user.id,
        request.form['hs_code'],
        request.form['start'],
        request.form['end'],
        request.form['currency']
    )


@app.route('/detail', methods=['POST'])
@login_required
def detail():
    country = request.form['country']
    currency = request.form['currency']

    df = analysis_result_by_user.get(current_user.id)

    if df is None or df.empty:
        return "데이터 없음"

    row = df[df['국가명'] == country].iloc[0]
    exchange_rate = load_exchange_rate()

    history = row.get('history', [])
    x = [pd.to_datetime(r['yyyymm'], format='%Y%m') for r in history]
    y = [r['unitPrice'] for r in history]

    if currency == "KRW":
        y = [v * exchange_rate for v in y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines', name=country,
        line=dict(width=4, color="#2563eb"), fill='tozeroy'
    ))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('detail.html', country=country, graphJSON=graphJSON, row=row)


# =========================
# 🩹 분석 기록 (신규)
# =========================
@app.route('/history')
@login_required
def history():
    records = AnalysisHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(AnalysisHistory.created_at.desc()).all()
    return render_template('history.html', records=records)


@app.route('/history/<int:history_id>')
@login_required
def history_detail(history_id):
    record = AnalysisHistory.query.get_or_404(history_id)

    if record.user_id != current_user.id:
        abort(403)  # 남의 기록은 못 봄

    df = pd.read_json(record.result_json, orient='records')
    return render_analysis_df(
        df, record.hs_code, record.start, record.end, record.currency,
        item_name=record.item_name
    )


if __name__ == '__main__':
    app.run(debug=True)
