from setuptools import setup, find_packages

# from distutils.core import setup

setup(
    name='peak_detection', # change to peak_detection once I have other packages
    # version='0.1.0',
    # packages=['RangingNN', 'Ionclassifier'],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # If you have non-Python files, specify them here:
        '': ['*.yaml', '*.pt', '*.csv']
    },
    url='https://github.com/wdwzyyg/peak_detection.git',
    license='MIT',
    author='Jingrui Wei',
    author_email='jwei74@wisc.edu',
    # description='',
    # keywords=[],
    install_requires=[
        "torch>=2.1.2, <=2.5.1", # keep at 2.1.2 for euler cluser
        "torchvision==0.16.2",
        "numpy>=1.24.1",
        "matplotlib",
        "scikit-image",
        "apav==1.4.0",
        "h5py",
        "pandas",
    ],

)

# peak_detection/
# ├── setup.py
# ├── peak_detection/
# │   ├── __init__.py
# │   └── RangingNN/
# │       ├── __init__.py
# │       └── ... (other files)
# setup(
#     name='peak_detection',
#     packages=['RangingNN'],  # Explicitly list the package
#     url='https://github.com/wdwzyyg/peak_detection.git',
# )