from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(5, 15)

    def on_start(self):
        self.client.get('/favicon.png', verify=False)
        self.client.get('/favicon.ico', verify=False)
        self.client.get('/manifest.json', verify=False)

    @task
    def index(self):
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
        res = self.client.get('/', verify=False)
        csrf_token = res.cookies['X-CSRF-Token']
        file = open('fixture/file.txt', 'rb')
        self.client.post(
            '/api/upload-file',
            headers={'X-CSRF-Token': csrf_token},
            files={'file': file},
            verify=False)
