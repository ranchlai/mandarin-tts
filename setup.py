import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtts", # Replace with your own username
    version="1.0.0",
    scripts=['mtts'],
    author="ranchlai",
    author_email="ranchlai@163.com",
    description="mandarin text to speech (TTS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranchlai/mandrian_tts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
