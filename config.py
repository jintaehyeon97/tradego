import os
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일에서 환경변수를 불러옵니다.
# .env는 git에 올라가지 않으며(.gitignore 처리), 실제 값은 .env.example을
# 참고해 각자 로컬에 직접 만들어야 합니다.
load_dotenv()

API_KEY = os.environ["API_KEY"]

# Flask 세션 암호화에 쓰이는 키
SECRET_KEY = os.environ["SECRET_KEY"]

# PostgreSQL 연결 정보
# postgresql+psycopg2://[사용자]:[비밀번호]@[호스트]:[포트]/[DB이름]
SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
