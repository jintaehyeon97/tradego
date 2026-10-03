from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    histories = db.relationship(
        "AnalysisHistory",
        backref="user",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(AnalysisHistory.created_at)",
    )


class AnalysisHistory(db.Model):
    __tablename__ = "analysis_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    hs_code = db.Column(db.String(20))
    item_name = db.Column(db.String(200))
    start = db.Column(db.String(6))
    end = db.Column(db.String(6))
    currency = db.Column(db.String(3))

    # 🩹 분석 결과(국가별 표 전체)를 JSON 문자열로 통째로 저장.
    # 이러면 기록을 다시 볼 때 API를 재호출하거나 재분석할 필요 없이
    # 그 시점 데이터 그대로 즉시 보여줄 수 있음.
    result_json = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
