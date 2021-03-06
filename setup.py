#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'daz', 'h5py']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'tensorflow']

coursera_requirements = ['matplotlib', 'scipy', 'Pillow', 'scikit-learn']

setup(
    author="Jean-Luc Thirion",
    author_email='thirion.jl@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="or machine learning centered around the derivation chain rule",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='chains',
    name='chains',
    packages=find_packages(include=['chains']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={
        'coursera': coursera_requirements,
    },
    url='https://github.com/thirionjl/chains',
    version='0.1.0',
    zip_safe=False,
)
