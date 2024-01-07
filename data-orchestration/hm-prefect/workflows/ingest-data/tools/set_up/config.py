from dotenv import load_dotenv
from utils.get_env_variable import get_env_variable

load_dotenv("tools/set_up/.env.production.local")

aws_default_region = get_env_variable("AWS_DEFAULT_REGION")
aws_access_key_id = get_env_variable("AWS_ACCESS_KEY_ID")
aws_secret_access_key = get_env_variable("AWS_SECRET_ACCESS_KEY")

flow_name = "ingest-data"
