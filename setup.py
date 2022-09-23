from setuptools import setup, find_packages

setup(
    name="annotator-metrics",
    version="0.0.1",
    packages=find_packages(),
    url="",
    license="",
    author="",
    author_email="",
    description="",
    install_requires=[
        "neuroglancer",
        "CNNectome @ git+https://github.com/saalfeldlab/CNNectome@feat/boundary-distance-metric",
    ],
)
