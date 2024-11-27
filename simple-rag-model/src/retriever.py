import torch
import torch.nn.functional as F
from sentence_transformers import util
from time import perf_counter as timer

class SemanticRetriever:
    def __init__(self, embeddings, chunks_data, similarity_metric='dot', min_score_threshold=0.1):
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embeddings must be a torch.Tensor")
        if not chunks_data:
            raise ValueError("chunks_data cannot be empty")
            
        self.embeddings = embeddings
        self.chunks_data = chunks_data
        self.similarity_metric = similarity_metric
        self.min_score_threshold = min_score_threshold
    
    def retrieve(self, query_embedding, top_k=5):
        """
        Retrieve relevant context for a query
        
        Args:
            query_embedding: Tensor of shape (embedding_dim,)
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing retrieved contexts
        """
        if not isinstance(query_embedding, torch.Tensor):
            raise ValueError("query_embedding must be a torch.Tensor")
            
        start_time = timer()
        
        try:
            # Calculate similarity scores based on metric
            if self.similarity_metric == 'dot':
                scores = util.dot_score(query_embedding, self.embeddings)[0]
            elif self.similarity_metric == 'cosine':
                scores = F.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    self.embeddings, 
                    dim=1
                )
            else:
                raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
                
            # Get top results
            top_results = torch.topk(scores, k=min(top_k, len(self.embeddings)))
            
            # Get indices and scores
            top_indices = top_results.indices.cpu().numpy()
            top_scores = top_results.values.cpu().numpy()
            
            # Build retrieved contexts
            retrieved_contexts = []
            filtered_count = 0
            
            for idx, score in zip(top_indices, top_scores):
                if score >= self.min_score_threshold:
                    context = self.chunks_data[idx]
                    retrieved_contexts.append({
                        'text': context['sentence_chunk'],
                        'similarity_score': float(score),
                        'page_number': context.get('page_num', None)
                    })
                else:
                    filtered_count += 1
            
            end_time = timer()
            print(f"Retrieval time: {end_time - start_time:.5f} seconds")
            print(f"Found {len(retrieved_contexts)} contexts above threshold {self.min_score_threshold}")
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} results below threshold")
            
            # If no results above threshold, return top result anyway
            if not retrieved_contexts and len(top_scores) > 0:
                context = self.chunks_data[top_indices[0]]
                retrieved_contexts.append({
                    'text': context['sentence_chunk'],
                    'similarity_score': float(top_scores[0]),
                    'page_number': context.get('page_num', None)
                })
                print("Returning top result despite being below threshold")
            
            return retrieved_contexts
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            raise
