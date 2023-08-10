import os

from dotenv import load_dotenv

load_dotenv(".env.production.local")

trino_host = os.getenv("TRINO_HOST")
trino_port = os.getenv("TRINO_PORT")
trino_user = os.getenv("TRINO_USER")
