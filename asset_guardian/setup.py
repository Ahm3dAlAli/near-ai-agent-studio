# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="defai-asset-guardian",
    version="0.1.0",
    author="DeFAI Team",
    author_email="info@defaiassetguardian.dev",
    description="Intelligent Portfolio Management on NEAR Blockchain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/defai-team/defai-asset-guardian",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "aiohttp>=3.9.1",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "click>=8.1.7",
        "near-api-py>=0.1.0",
        "openai>=1.6.1",
        "pandas>=2.0.0",
        "colorlog>=6.8.0",
        "prompt_toolkit>=3.0.43",
    ],
    entry_points={
        "console_scripts": [
            "defai-guardian=src.app:main",
        ],
    },
)
