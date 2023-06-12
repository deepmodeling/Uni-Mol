"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="unimol_tools",
    version="1.0.0",
    description=("unimol_tools is a Python package for property prediciton with Uni-Mol in molecule, materials and protein."),
    author="DP Technology",
    author_email="unimol@dp.tech",
    license="The MIT License",
    url="https://github.com/dptech-corp/Uni-Mol",
    packages=find_packages(
        where='.',
        exclude=[
            "build",
            "dist",
        ],
    ),
    install_requires=["yacs", "addict", "tqdm", "transformers", "pymatgen"],
    python_requires=">=3.6",
    include_package_data=True,
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