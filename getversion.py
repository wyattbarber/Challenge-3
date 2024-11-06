import tomllib
import os
import sys


def main(path: str):
    with open(os.path.join(path, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
        version = data["project"]["version"]
    print(version)


if __name__ == "__main__":
    main(sys.argv[1])