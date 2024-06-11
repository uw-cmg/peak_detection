from setuptools import setup, find_packages

# from distutils.core import setup

setup(
    name='peak_detection',
    # version='0.1.0',
    packages=find_packages(),
    url='https://github.com/wdwzyyg/peak_detection.git',
    license='MIT',
    author='Jingrui Wei',
    author_email='jwei74@wisc.edu',
    # description='',
    # keywords=[],
    install_requires=[
        "torch>=2.3.1",
        "numpy>=1.24.1",
        "matplotlib==3.7.1",
        "scikit-image",
    ],

)
