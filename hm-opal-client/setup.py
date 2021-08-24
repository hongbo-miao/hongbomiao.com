import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    requirements = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name="hm-opal-fetcher-postgres",
    version="1.0.0",
    python_requires=">=3.8",
    install_requires=requirements,
    packages=find_packages(),
)
