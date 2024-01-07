import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path("../api-node") / ".env.development.local"
load_dotenv(env_path)

seed_user_email = os.getenv("SEED_USER_EMAIL")
seed_user_password = os.getenv("SEED_USER_PASSWORD")
