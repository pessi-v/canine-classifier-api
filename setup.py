# setup.py
from setuptools import setup, find_packages

with open("requirements_prod.txt") as f:
    content = f.readlines()
    requirements = [x.strip() for x in content if "git+" not in x]

setup(name='snoop_dog',
      version="0.0.1",
      description="A FastAPI using a finetunded EfficientNet V2s model to classify dog breeds",
      packages=find_packages(),
      install_requires=requirements
)
