#!/usr/bin/env python

"""The setup script."""
from os import path
from setuptools import setup, find_packages
import sys
import versioneer

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
    author="Ronald J Pandolfi",
    author_email='ronpandolfi@lbl.gov',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="A file exporter for Bluesky's Databroker.",
    entry_points={
        'console_scripts': [
            'bluesky_exporter=bluesky_exporter:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='bluesky_exporter',
    name='bluesky_exporter',
    packages=find_packages(include=['bluesky_exporter', 'bluesky_exporter.*']),
    test_suite='tests',
    url='https://github.com/ronpandolfi/bluesky_exporter',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs']
    }
)
