import json
import os

import config
import magic
from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(5, 15)
    jwt_token = ""

    def on_start(self) -> None:
        res = self.client.get("/", verify=False)

        # Get JWT token
        query = """
            mutation SignIn($email: String!, $password: String!) {
                signIn(email: $email, password: $password) {
                    jwtToken
                }
            }
        """
        variables = {
            "email": config.seed_user_email,
            "password": config.seed_user_password,
        }
        res = self.client.post(
            "/graphql",
            json={"query": query, "variables": variables},
        )
        content = json.loads(res.content)
        self.jwt_token = content["data"]["signIn"]["jwtToken"]

    @task
    def index(self) -> None:
        self.client.get("/", verify=False)

    @task
    def fetch_me(self) -> None:
        query = """
            query Me {
                me {
                    id
                    name
                    bio
                }
            }
        """
        self.client.post("/graphql", json={"query": query}, verify=False)

    @task
    def upload_file(self) -> None:
        file_path = "src/fixture/image.png"
        file_name = os.path.basename(file_path)
        file = open(file_path, "rb")
        mime = magic.Magic(mime=True)
        file_mimetype = mime.from_file(file_path)
        files = {
            "file": (file_name, file, file_mimetype),
        }
        self.client.post(
            "/api/upload-file",
            headers={
                "Authorization": f"Bearer {self.jwt_token}",
            },
            files=files,
            verify=False,
        )
