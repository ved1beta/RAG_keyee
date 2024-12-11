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
        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_len].to_dict(orient="records")
     
        text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
        
        for item in tqdm(pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        
        text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        return text_chunks_and_embeddings_df

    def save_embeddings(self, embeddings_df, save_path):
    
        embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: x.tolist())
       
        if save_path.endswith('.csv'):
            embeddings_df['embedding'] = embeddings_df['embedding'].apply(json.dumps)
            embeddings_df.to_csv(save_path, index=False)
        else:
            raise ValueError("Save path must end with .csv ")

    def load_embeddings(self, load_path):
        if load_path.endswith('.csv'):
            df = pd.read_csv(load_path)
            df['embedding'] = df['embedding'].apply(json.loads)
        else:
            raise ValueError("Load path must end with .csv")
        
        df['embedding'] = df['embedding'].apply(np.array)
        
        embeddings = torch.tensor(
            np.array(df['embedding'].tolist()), 
            dtype=torch.float32
        ).to(self.device)
        
        return df, embeddings
