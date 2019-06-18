import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynto-punkbrwstr",
    version="0.0.1",
    author="Peter Graf",
    author_email="magnumpi@gmail.com",
    description="Time series analysis using a concatenative paradigm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/punkbrwstr/pynto",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)