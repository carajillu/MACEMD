from setuptools import setup, find_packages
setup(
    name="NaiMLeSS",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "ase",
        "torch",
        "mace",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "naimless=naimless.main:main",
        ],
    },
)