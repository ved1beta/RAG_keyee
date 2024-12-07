import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
from retriever import SemanticRetriever

class QueryProcessor:
    def __init__(self, 
                 embeddings_path="text_chunks_and_embeddings_df.csv", 
                 model_name="all-mpnet-base-v2", 
                 device=None,
                 similarity_metric='dot',
                 min_score_threshold=0.1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embeddings and chunks
        self.text_chunks_and_embedding_df = self._load_embeddings(embeddings_path)
        print(f"Loaded {len(self.text_chunks_and_embedding_df)} text chunks")
        
        self.pages_and_chunks = self.text_chunks_and_embedding_df.to_dict(orient="records")
        self.embeddings = torch.tensor(
            np.array(self.text_chunks_and_embedding_df["embedding"].tolist()), 
            dtype=torch.float32
        ).to(self.device)
        
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name, 
            device=self.device
        )
        
        # Initialize retriever
        self.retriever = SemanticRetriever(
            embeddings=self.embeddings,
            chunks_data=self.pages_and_chunks,
            similarity_metric=similarity_metric,
            min_score_threshold=min_score_threshold
        )

    def _load_embeddings(self, embeddings_path):
        df = pd.read_csv(embeddings_path)
        df["embedding"] = df["embedding"].apply(
            lambda x: np.array(json.loads(x))
        )
        return df

    def encode_query(self, query):
        return self.embedding_model.encode(query, convert_to_tensor=True)

    def process_query(self, query, k=5):
        """Process a query and return relevant contexts"""
        try:
            # Encode query
            query_embedding = self.encode_query(query)
            
            # Retrieve contexts using the retriever
            retrieved_contexts = self.retriever.retrieve(
                query_embedding=query_embedding,
                top_k=k
            )
            
            if not retrieved_contexts:
                print("No contexts found for query")
                return []
            
            print(f"Found {len(retrieved_contexts)} relevant contexts")
            return retrieved_contexts
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return []