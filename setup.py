from setuptools import setup, PEP420PackageFinder

with open('requirements.txt', 'r') as infile:
    requirements = [line.strip() for line in infile.readlines()]

setup(
    name='toy_proc',
    description='toy_proc',
    author='hughperkins@gmail.com',
    packages=PEP420PackageFinder.find(exclude=('test*',)),
    install_requires=requirements,
    include_package_data=True
)
