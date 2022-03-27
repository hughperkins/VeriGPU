from setuptools import setup, PEP420PackageFinder

with open('requirements.txt', 'r') as infile:
    requirements = [line.strip() for line in infile.readlines()]

setup(
    name='verigpu',
    description='verigpu',
    author='hughperkins@gmail.com',
    packages=PEP420PackageFinder.find(exclude=('test*',)),
    install_requires=requirements,
    include_package_data=True
)
