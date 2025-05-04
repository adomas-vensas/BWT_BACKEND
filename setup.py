from setuptools import setup, find_packages

setup(
    name="bwt",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
    ],
    python_requires=">=3.8",
)