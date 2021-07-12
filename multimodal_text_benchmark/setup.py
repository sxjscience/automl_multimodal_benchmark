#!/usr/bin/env python
from datetime import datetime
import io
import os
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('src', 'auto_mm_bench', '__init__.py')

if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')

requirements = [
    'absl-py',
    'boto3',
    'javalang>=0.13.0',
    'h5py>=2.10.0',
    'yacs>=0.1.8',
    'protobuf',
    'unidiff',
    'sentencepiece',
    'tqdm',
    'xarray',
    'regex',
    'requests',
    'jsonlines',
    'contextvars',
    'pyarrow',
    'pandas',
    'py-cpuinfo',
    'contextvars;python_version<"3.7"',  # Contextvars for python <= 3.6
    'dataclasses;python_version<"3.7"',  # Dataclass for python <= 3.6
    'fasttext>=0.9.1,!=0.9.2'  # Fix to 0.9.1 due to https://github.com/facebookresearch/fastText/issues/1052
]

setup(
    # Metadata
    name='auto_mm_bench',
    version=VERSION,
    python_requires='>=3.6',
    description='Benchmarks Multimodal AutoML on Structured Data Tables with Text, Categorical, and Numerical Data',
    long_description_content_type='text/markdown',
    license='Apache-2.0',

    # Package info
    packages=find_packages(where="src", exclude=(
        'tests',
        'scripts',
    )),
    package_dir={"": "src"},
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
)
