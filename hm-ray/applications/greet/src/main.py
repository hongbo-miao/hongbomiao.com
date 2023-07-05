import ray


@ray.remote
def greet():
    return "Hello, World!"


if __name__ == "__main__":
    ray.init()
    print(ray.get(greet.remote()))
