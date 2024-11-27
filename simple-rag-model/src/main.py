import logging
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
