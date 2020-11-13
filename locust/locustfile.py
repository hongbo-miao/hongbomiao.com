from locust import HttpUser, between, task
import json

import config


class WebsiteUser(HttpUser):
    wait_time = between(5, 15)

    csrf_token = ""
    jwt_token = ""

    def on_start(self):
        # Get CSRF token
        res = self.client.get('/', verify=False)
        self.csrf_token = res.cookies['__Host-csrfToken']

        # Get JWT token
        query = """
            mutation SignIn($email: String!, $password: String!) {
                signIn(email: $email, password: $password) {
                    jwtToken
                }
            }
        """
        variables = {'email': config.seed_user_email, 'password': config.seed_user_password}
        res = self.client.post('/graphql', json={'query': query, 'variables': variables})
        content = json.loads(res.content)
        self.jwt_token = content['data']['signIn']['jwtToken']

    @task
    def index(self):
        self.client.get('/', verify=False)

    @task
    def fetch_me(self):
        query = """
            query Me {
                me {
                    id
                    name
                    bio
                }
            }
        """
        self.client.post('/graphql', json={'query': query}, verify=False)

    @task
    def upload_file(self):
        file = open('fixture/file.txt', 'rb')
        self.client.post(
            '/api/upload-file',
            headers={'Authorization': f'Bearer {self.jwt_token}', 'X-CSRF-Token': self.csrf_token},
            files={'file': file},
            verify=False)
