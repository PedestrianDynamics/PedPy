import setuptools

import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pita",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="T. Schroedter",
    description="PiTA is a python library to analyze trajectories of pedestrians.",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Shapely >= 1.8.1",
        "pygeos==0.12.0",
        "scipy==1.8.0",
    ],
    entry_points={"console_scripts": ["pita=analyzer.application:main"]},
)
