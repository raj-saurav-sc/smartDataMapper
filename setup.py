import setuptools
import os

# Function to read the long description from the README file
def read_me(file_name="README.md"):
    with open(os.path.join(os.path.dirname(__file__), file_name), "r", encoding="utf-8") as fh:
        return fh.read()

# List of dependencies
install_requires = [
    "pandas",
    "numpy",
    "sentence-transformers",
    "scikit-learn",
    "colorama",
    "lxml",
    "python-levenshtein",
    "torch",
]

setuptools.setup(
    name="smart-data-mapper",
    version="0.1.0",
    author="Smart Data Mapper Team",
    author_email="<your-email@example.com>", # Placeholder
    description="A Python tool for intelligent, data-aware schema mapping suggestions.",
    long_description=read_me(),
    long_description_content_type="text/markdown",
    url="https://github.com/raj-saurav-sc/smartDataMapper",
    py_modules=[
        'smartautoMapper',
        'report_generator',
        'visualize_mappings',
        'etl_generator'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Intended Audience :: Developers",
        "Environment :: Console",
    ],
    python_requires='>=3.9',
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'smart-mapper=smartautoMapper:main',
        ],
    },
)