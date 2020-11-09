from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
  wait_time = between(5, 15)

  def on_start(self):
    self.client.get('favicon.png', verify=False)
    self.client.get('favicon.ico', verify=False)
    self.client.get('manifest.json', verify=False)

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
    self.client.post('graphql', json={'query': query}, verify=False)
