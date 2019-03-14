from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyPho',
    version='0.0.1',
    description=("A photonics package for Python."),
    long_description=readme,
    author='Dominik S. Wild',
    author_email='dominikswild@gmail.com',
    url='https://github.com/dominikswild/pypho',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
