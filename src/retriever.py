import torch
import torch.nn.functional as F
from sentence_transformers import util
from time import perf_counter as timer

class SemanticRetriever:
    def __init__(self, embeddings, chunks_data, similarity_metric='cosine', min_score_threshold=0.1):
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embeddings must be a torch.Tensor")
        if not chunks_data:
            raise ValueError("chunks_data cannot be empty")
            
        self.embeddings = embeddings
        self.chunks_data = chunks_data
        self.similarity_metric = similarity_metric.lower()  # Convert to lowercase
        self.min_score_threshold = min_score_threshold
        
        # Validate similarity metric
        if self.similarity_metric not in ['dot', 'cosine']:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}. Use 'dot' or 'cosine'.")

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
            # Ensure query_embedding is 2D for cosine similarity
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
                
            # Calculate similarity scores based on metric
            if self.similarity_metric == 'dot':
                scores = util.dot_score(query_embedding, self.embeddings)[0]
            else:  # cosine similarity
                # Normalize embeddings for cosine similarity
                query_embedding_normalized = F.normalize(query_embedding, p=2, dim=1)
                embeddings_normalized = F.normalize(self.embeddings, p=2, dim=1)
                scores = torch.mm(query_embedding_normalized, embeddings_normalized.t())[0]
            
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
                    # Check if 'sentence_chunk' exists in context, if not try 'text' or use the entire context
                    text = context.get('sentence_chunk', context.get('text', str(context)))
                    retrieved_contexts.append({
                        'text': text,
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
            
            # Always return at least one result if available
            if not retrieved_contexts and len(top_scores) > 0:
                context = self.chunks_data[top_indices[0]]
                text = context.get('sentence_chunk', context.get('text', str(context)))
                retrieved_contexts.append({
                    'text': text,
                    'similarity_score': float(top_scores[0]),
                    'page_number': context.get('page_num', None)
                })
                print("Returning top result despite being below threshold")
            
            # Print debug info
            print(f"Debug - Retrieved contexts count: {len(retrieved_contexts)}")
            if retrieved_contexts:
                print(f"Debug - First context preview: {retrieved_contexts[0]['text'][:100]}...")
            
            return retrieved_contexts
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            # Return a default context in case of error
            return [{
                'text': "Error occurred during retrieval. Please try again.",
                'similarity_score': 0.0,
                'page_number': None
            }]