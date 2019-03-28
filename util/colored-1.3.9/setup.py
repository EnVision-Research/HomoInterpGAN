#!/usr/bin/python
# -*- coding: utf-8 -*-


from distutils.core import setup
from colored import __version__


setup(
    name="colored",
    packages=["colored"],
    version=__version__,
    description="Simple library for color and formatting to terminal",
    keywords=["color", "colour", "paint", "ansi", "terminal", "linux",
              "python"],
    author="dslackw",
    author_email="d.zlatanidis@gamil.com",
    url="https://gitlab.com/dslackw/colored",
    package_data={"": ["LICENSE", "README.rst", "CHANGELOG"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Operating System :: POSIX :: Other",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Unix Shell",
        "Topic :: Terminals"],
    long_description=open("README.rst").read()
)
