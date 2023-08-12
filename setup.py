# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:22:16 2023

@author: kevin
"""

from setuptools import setup, find_packages
import os

# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Kevin's firt setup file

def read_file(file):
   with open(file) as f:
        return f.read()
    
long_description = read_file("README.md")

setup(
    name = 'GLM_RNN',
    version = '1.0.0',
    author = 'Kevin S. Chen',
    author_email = 'kschen@princeton.edu',
    url = 'https://github.com/Kevin-Sean-Chen/GLM_RNN',
    description = 'Code for GLM-RNN to explore network dynamics and inference.',
    long_description_content_type = "text/x-rst",  # If this causes a warning, upgrade your setuptools package
    long_description = long_description,
    license = "MIT license",
    packages = find_packages(exclude=["test"]),  # Don't include test directory in binary distribution
#    install_requires = requirements,
    py_modules=['glmrnn.glmrnn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]  # Update these accordingly
)