import torch
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer

class QueryProcessor:
    def __init__(self, 
                 embeddings_path="text_chunks_and_embeddings_df.csv", 
                 model_name="all-mpnet-base-v2", 
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.text_chunks_and_embedding_df = self._load_embeddings(embeddings_path)
        
        self.pages_and_chunks = self.text_chunks_and_embedding_df.to_dict(orient="records")
        self.embeddings = torch.tensor(
            np.array(self.text_chunks_and_embedding_df["embedding"].tolist()), 
            dtype=torch.float32
        ).to(self.device)
        
        self.embedding_model = SentenceTransformer(
            model_name_or_path=model_name, 
            device=self.device
        )

    def _load_embeddings(self, embeddings_path):
        df = pd.read_csv(embeddings_path)
        df["embedding"] = df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ")
        )
        return df

    def encode_query(self, query):
        return self.embedding_model.encode(query, convert_to_tensor=True)

    def get_similarity_scores(self, query_embedding):
        start_time = timer()
        dot_scores = util.dot_score(a=query_embedding, b=self.embeddings)[0]
        end_time = timer()
        
        print(f"Time taken to get scores on {len(self.embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
        
        return dot_scores

    def get_top_k_results(self, dot_scores, k=5):
        top_results = torch.topk(dot_scores, k=k)
        return top_results

    def retrieve_context(self, top_results):
        top_indices = top_results.indices.cpu().numpy()
        top_scores = top_results.values.cpu().numpy()
        
        retrieved_contexts = []
        for idx, score in zip(top_indices, top_scores):
            context = self.pages_and_chunks[idx]
            retrieved_contexts.append({
                'text': context['sentence_chunk'],
                'similarity_score': score
            })
        
        return retrieved_contexts

    def process_query(self, query, k=5):
        query_embedding = self.encode_query(query)
        
        dot_scores = self.get_similarity_scores(query_embedding)
        
        top_results = self.get_top_k_results(dot_scores, k)
        
        retrieved_contexts = self.retrieve_context(top_results)
        
        return retrieved_contexts