import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtts",  # Replace with your own username
    version="1.0.0",
    author="ranchlai",
    author_email="",
    description="Mandarin text to speech (mtts)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranchlai/mandrian_tts",
    packages=setuptools.find_packages(
        exclude=["build*", "test*", "examples*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch >= 1.5.0',
        'tqdm ~= 4.49.0',
        'librosa~=0.8.0',
        'numpy >= 1.15.0', 
        'scipy >= 1.0.0', 
        'resampy >= 0.2.2',
        'soundfile >= 0.9.0',
        'PyYAML == 5.4.1',
        'tensorboard == 2.5.0',
        'matplotlib==3.3.4',
        'pypinyin==0.41.0'
    ],
   
    
)
