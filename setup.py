from setuptools import setup, find_packages

setup(
    name="one_ring",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "loguru",
    ],
)
