"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="unimol_tools",
    version="0.1.1",
    description=("unimol_tools is a Python package for property prediciton with Uni-Mol in molecule, materials and protein."),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="DP Technology",
    author_email="unimol@dp.tech",
    license="The MIT License",
    url="https://github.com/deepmodeling/Uni-Mol/unimol_tools",
    packages=find_packages(
        where='.',
        exclude=[
            "build",
            "dist",
        ],
    ),
    install_requires=["numpy<2.0.0,>=1.22.4",
                      "pandas<2.0.0",
                      "torch",
                      "joblib",
                      "rdkit",
                      "pyyaml",
                      "addict",
                      "scikit-learn",
                      "tqdm"],
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