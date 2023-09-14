"""Properties for creating wheels to be distributed."""
# type: ignore
import setuptools

import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PedPy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="T. Schrödter",
    description="PedPy is a Python module for pedestrian movement analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    packages=[
        "pedpy",
        "pedpy.data",
        "pedpy.io",
        "pedpy.methods",
        "pedpy.plotting",
    ],
    python_requires=">=3.10",
    install_requires=[
        "aenum~=3.1",
        "numpy~=1.25",
        "pandas~=2.0",
        "Shapely~=2.0",
        "scipy~=1.11",
        "matplotlib~=3.7",
    ],
)
