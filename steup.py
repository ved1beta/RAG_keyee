from setuptools import setup, find_packages

setup(
    name='simple-rag-model',
    version='0.1.0',
    description='A simple Retrieval-Augmented Generation (RAG) model project',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Core ML and NLP libraries
        'torch>=1.10.0',
        'transformers>=4.30.0',
        'sentence-transformers>=2.2.0',
        
        # Data processing and storage
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'pyarrow>=7.0.0',  # Efficient data storage
        
        # Vector database and retrieval
        'faiss-cpu>=1.7.2',  # Or faiss-gpu if using GPU
        
        # Utility libraries
        'tqdm>=4.62.0',  # Progress bars
        'pyyaml>=5.4.1',  # Configuration management
        
        # Logging and debugging
        'loguru>=0.5.3',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'mypy>=0.910',
            'black>=21.8b0',
            'isort>=5.9.3',
        ],
        'gpu': [
            'torch>=1.10.0',
            'faiss-gpu>=1.7.2',
        ]
    },
    entry_points={
        'console_scripts': [
            'rag-train=src.main:train',
            'rag-inference=src.main:inference',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)