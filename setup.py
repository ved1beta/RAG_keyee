from setuptools import setup, find_packages

setup(
    name='rag-model',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        'pyyaml'
    ],
    extras_require={
        'dev': ['pytest', 'black']
    }
)
