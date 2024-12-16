import os


def get_boolean_env_variable(name: str) -> bool:
    return os.getenv(name, "False").lower() in ["1", "true"]
