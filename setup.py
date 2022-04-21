from distutils.util import convert_path

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

main_ns = {}
ver_path = convert_path("report/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setuptools.setup(
    name="jpsreport",
    version=main_ns["__version__"],
    author="T. Schroedter",
    author_email="author@example.com",
    description="JPSreport is a command line module to analyze trajectories of pedestrians.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/schroedtert/jpsreport-python",
    project_urls={
        "Bug Tracker": "https://github.com/schroedtert/jpsreport-python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "aenum >= 3.1.5",
        "numpy >= 1.21.4",
        "pandas >= 1.3.4",
        "Shapely == 1.8.0",
    ],
    entry_points={"console_scripts": [" jpsreport=report.application:main"]},
)
