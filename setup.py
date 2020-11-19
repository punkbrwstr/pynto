import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynto",
    version="0.2",
    author="Peter Graf",
    author_email="peter@pynto.tech",
    description="Data analysis using a concatenative paradigm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/punkbrwstr/pynto",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='data analysis quantitative tabular concatenative functional',
    python_requires='>=3.6',
    install_requires=['numpy','pandas','python-dateutil', 'timestamps'],
)
