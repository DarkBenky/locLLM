def hello(name):
    return f"Hello {name}"


class Greeter:
    def greet(self):
        return hello("world")


async def fetch_data():
    return "data"
