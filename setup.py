#!/usr/bin/env python
from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read()
setup(
    # Metadata
    name="hugnlp",
    version="0.0.1",
    python_requires=">=3.6",
    author="@Jianing Wang @HugAILab",
    author_email="lygwjn@gmail.com",
    url="https://github.com/wjn1996",
    description="HugNLP Library",
    long_description=readme,
    entry_points={"console_scripts": ["hugnlp=hugnlp.cli:main"]},
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="Apache-2.0",

    #Package info
    install_requires=requirements)
