import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DistSup-distsup", # Replace with your own username
    version="0.0.1",
    author="Gistsup JSALT 2019 Team",
    author_email="jch@cs.uni.wroc.pl",
    description="Distant supervision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/distsup/DistSup",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
