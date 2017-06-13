from setuptools import setup, find_packages

setup(
    name = "pdnn",
    author = "Peter O'Connor",
    author_email = "poconn4@gmail.com",
    version = 0,
    dependency_links = (),
    install_requires = ['numpy', 'matplotlib', 'theano', 'scipy'],
    scripts = [],
    packages=find_packages(),
    )
