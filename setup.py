from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="interview-prep",
    version="0.1.0",
    author="Karan Kumar",
    author_email="karankum451@gmail.com",
    description="Interview preparation project",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9",
)
