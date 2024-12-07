import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json

class EmbeddingGenerator:
    def __init__(self, model_name="all-mpnet-base-v2", device="cuda"):
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_embeddings(self, df, min_token_len=5, batch_size=32):
        # Filter chunks based on token length
        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_len].to_dict(orient="records")
        
        # Extract text chunks
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
        
        # Generate embeddings
        for item in tqdm(pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        
        # Create DataFrame with chunks and embeddings
        text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        
        return text_chunks_and_embeddings_df

    def save_embeddings(self, embeddings_df, save_path):
        """
        Save embeddings to file with proper serialization
        """
        # Convert numpy arrays to lists for serialization
        embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: x.tolist())
        
        # Save to file based on extension
        if save_path.endswith('.csv'):
            # Convert embeddings to JSON string for CSV storage
            embeddings_df['embedding'] = embeddings_df['embedding'].apply(json.dumps)
            embeddings_df.to_csv(save_path, index=False)
        elif save_path.endswith('.parquet'):
            # Parquet handles lists natively
            embeddings_df.to_parquet(save_path, index=False)
        else:
            raise ValueError("Save path must end with .csv or .parquet")

    def load_embeddings(self, load_path):
        """
        Load embeddings from file with proper deserialization
        """
        if load_path.endswith('.csv'):
            # Load CSV and parse JSON embeddings
            df = pd.read_csv(load_path)
            df['embedding'] = df['embedding'].apply(json.loads)
        elif load_path.endswith('.parquet'):
            # Load parquet file
            df = pd.read_parquet(load_path)
        else:
            raise ValueError("Load path must end with .csv or .parquet")
        
        # Convert embeddings to numpy arrays
        df['embedding'] = df['embedding'].apply(np.array)
        
        # Convert embeddings to torch tensor
        embeddings = torch.tensor(
            np.array(df['embedding'].tolist()), 
            dtype=torch.float32
        ).to(self.device)
        
        return df, embeddings