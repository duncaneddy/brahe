#!/usr/bin/env python3
'''
brahe setup config
'''

import pathlib
from setuptools import setup, find_packages

# Read package description
about = {}
with open(pathlib.Path.cwd() / 'brahe' / '__about__.py') as fobj:
    exec(fobj.read(), about)

# Read common requirements
requirements = []
with open(pathlib.Path.cwd() / 'requirements.txt') as fp:
    for line in fp:
        requirements.append(line.strip())

setup(
    name='brahe',
    version=about['__version__'],
    packages = find_packages(),
    install_requires = requirements,
    include_package_data = True,
    author               = about['__author__'],
    author_email         = about['__author_email__'],
    maintainer           = about['__author__'],
    maintainer_email     = about['__author_email__'],
    description          = about['__description__'],
    url                  = about['__url__'],
    entry_points={
        'console_scripts': [
            'brahe = brahe.cli.cli:cli_group',
        ],
    }
)