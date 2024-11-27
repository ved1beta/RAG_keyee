import os
import shutil
import argparse

class RAGProjectGenerator:
    def __init__(self, project_name='simple-rag-model'):
        """
        Initialize RAG project generator
        
        Args:
            project_name (str): Name of the project to be created
        """
        self.project_name = project_name
        self.base_dir = os.path.abspath(project_name)
        
        # Project structure
        self.directories = [
            'src',
            'configs',
            'data/raw',
            'data/processed',
            'tests',
            'notebooks',
            'scripts'
        ]
        
        # Template files with their content and paths
        self.files = [
            {
                'path': os.path.join('src', '__init__.py'),
                'content': '# RAG Model Source Package\n'
            },
            {
                'path': os.path.join('src', 'main.py'),
                'content': self._get_main_py_template()
            },
            {
                'path': os.path.join('src', 'data_processing.py'),
                'content': self._get_data_processing_template()
            },
            {
                'path': os.path.join('src', 'model.py'),
                'content': self._get_model_py_template()
            },
            {
                'path': os.path.join('src', 'retriever.py'),
                'content': self._get_retriever_py_template()
            },
            {
                'path': os.path.join('src', 'utils.py'),
                'content': self._get_utils_py_template()
            },
            {
                'path': os.path.join('configs', '__init__.py'),
                'content': '# Configuration Package\n'
            },
            {
                'path': os.path.join('configs', 'config.yaml'),
                'content': self._get_config_yaml_template()
            },
            {
                'path': os.path.join('tests', '__init__.py'),
                'content': '# RAG Model Tests\n'
            },
            {
                'path': 'README.md',
                'content': self._get_readme_template()
            },
            {
                'path': 'setup.py',
                'content': self._get_setup_py_template()
            },
            {
                'path': os.path.join('notebooks', 'exploration.ipynb'),
                'content': self._get_notebook_template()
            },
            {
                'path': os.path.join('scripts', 'train.py'),
                'content': self._get_train_script_template()
            },
            {
                'path': os.path.join('scripts', 'inference.py'),
                'content': self._get_inference_script_template()
            }
        ]
    
    def generate(self):
        """
        Generate the entire project structure
        """
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories
        for directory in self.directories:
            os.makedirs(os.path.join(self.base_dir, directory), exist_ok=True)
        
        # Create files
        for file_info in self.files:
            full_path = os.path.join(self.base_dir, file_info['path'])
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write file
            with open(full_path, 'w') as f:
                f.write(file_info['content'])
        
        print(f"Project '{self.project_name}' generated successfully!")
    
    # Template generation methods
    def _get_main_py_template(self):
        return '''import logging
import argparse
from typing import Dict, Any

def train(config_path: str):
    """Main training function"""
    logging.info(f"Training with config: {config_path}")

def inference(config_path: str, query: str):
    """Inference function"""
    logging.info(f"Inferencing query: {query}")

def main():
    parser = argparse.ArgumentParser(description="RAG Model")
    parser.add_argument('--mode', choices=['train', 'inference'], required=True)
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--query', help='Query for inference')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'inference':
        if not args.query:
            parser.error("Query is required for inference")
        inference(args.config, args.query)

if __name__ == '__main__':
    main()
'''
    
    def _get_data_processing_template(self):
        return '''import pandas as pd
from typing import Dict, Any

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def load_data(self, split='train'):
        """Load and preprocess data"""
        # Implement data loading logic
        pass
    
    def preprocess(self, data):
        """Preprocess data"""
        # Implement preprocessing steps
        return data
'''
    
    def _get_model_py_template(self):
        return '''class RAGModel:
    def __init__(self, config):
        self.config = config
    
    def train(self, train_data, val_data):
        """Train the RAG model"""
        pass
    
    def generate(self, query, context):
        """Generate response for given query and context"""
        pass
'''
    
    def _get_retriever_py_template(self):
        return '''class SemanticRetriever:
    def __init__(self, config):
        self.config = config
    
    def retrieve(self, query, top_k=5):
        """Retrieve relevant context for a query"""
        pass
'''
    
    def _get_utils_py_template(self):
        return '''import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
'''
    
    def _get_config_yaml_template(self):
        return '''# RAG Model Configuration

data:
  path: 'data/raw'
  
model:
  type: 'transformer'
  name: 'facebook/bart-base'

retrieval:
  top_k: 5
  
training:
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
'''
    
    def _get_readme_template(self):
        return '''# RAG Model Project

## Setup
```bash
pip install -e .
```

## Training
```bash
python src/main.py --mode train
```

## Inference
```bash
python src/main.py --mode inference --query "Your query here"
```
'''
    
    def _get_setup_py_template(self):
        return '''from setuptools import setup, find_packages

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
'''
    
    def _get_gitignore_template(self):
        return '''# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.env/

# Jupyter
.ipynb_checkpoints/

# Model and data
*.pt
*.pth
data/processed/

# IDE
.vscode/
.idea/
'''
    
    def _get_notebook_template(self):
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# RAG Model Exploration"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["# Initial data and model exploration"]
  }
 ]
}
'''
    
    def _get_train_script_template(self):
        return '''#!/usr/bin/env python
from src.main import train

if __name__ == '__main__':
    train('configs/config.yaml')
'''
    
    def _get_inference_script_template(self):
        return '''#!/usr/bin/env python
from src.main import inference

if __name__ == '__main__':
    inference('configs/config.yaml', "Sample query")
'''

def main():
    parser = argparse.ArgumentParser(description="RAG Project Generator")
    parser.add_argument(
        '--name', 
        default='simple-rag-model', 
        help='Name of the project to generate'
    )
    
    args = parser.parse_args()
    
    # Generate project
    generator = RAGProjectGenerator(args.name)
    generator.generate()

if __name__ == '__main__':
    main()