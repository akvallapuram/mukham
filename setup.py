import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = list(map(lambda x: x.replace('==', '>='), fh.read().split('\n')))

setuptools.setup(
    name="mukham",
    version="1.0.1",
    author="Anish Krishna Vallapuram",
    author_email="akvallapuram@connect.ust.hk",
    description="A basic library to crop largest face from the images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akvallapuram/mukham",
    packages=['mukham'],
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)