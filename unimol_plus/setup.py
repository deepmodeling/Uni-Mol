"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="unimol_plus",
    version="1.0.0",
    description="",
    author="DP Technology",
    author_email="unimol@dp.tech",
    license="The MIT License",
    url="https://github.com/dptech-corp/Uni-Mol",
    packages=find_packages(
        exclude=["scripts", "tests", "example_data", "docker", "figure"]
    ),
    install_requires=[
        "numpy",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
