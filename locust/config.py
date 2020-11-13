from dotenv import load_dotenv
from pathlib import Path
import os


env_path = Path('../server') / '.env.development.local'
load_dotenv(dotenv_path=env_path)

seed_user_email = os.getenv('SEED_USER_EMAIL')
seed_user_password = os.getenv('SEED_USER_PASSWORD')
