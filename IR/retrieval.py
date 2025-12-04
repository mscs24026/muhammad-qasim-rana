from preprocess import preprocess
import numpy as np

class RetrievalEngine:
    def __init__(self, indexer):
        """
        Initialize RetrievalEngine with an Indexer instance.
        
        Args:
            indexer: An Indexer object containing TF-IDF and BM25 models
        """
        self.indexer = indexer
    
    def tfidf_search(self, query, top_k=5):
        """
        Perform TF-IDF based search on the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of tuples (doc_id, score) for top_k documents
        """
        if not query or not query.strip():
            return []
        
        
        query_tok = " ".join(preprocess(query))
        
        
        if not hasattr(self.indexer, 'tfidf_vectorizer') or not hasattr(self.indexer, 'tfidf_matrix'):
            raise AttributeError("Indexer does not have TF-IDF models built. Call build_indexes() first.")
        
        
        query_vec = self.indexer.tfidf_vectorizer.transform([query_tok])
        
        
        scores = (query_vec @ self.indexer.tfidf_matrix.T).toarray()[0]
        
        
        n_docs = len(scores)
        k = min(top_k, n_docs)
        top_idx = np.argsort(scores)[::-1][:k]
        
        
        results = [(self.indexer.doc_ids[i], float(scores[i])) 
                  for i in top_idx if scores[i] > 0]
        
        return results[:top_k]  
    
    def bm25_search(self, query, top_k=5):
        """
        Perform BM25 based search on the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of tuples (doc_id, score) for top_k documents
        """
        if not query or not query.strip():
            return []
        
        
        query_tok = preprocess(query)
        
        
        if not hasattr(self.indexer, 'bm25'):
            raise AttributeError("Indexer does not have BM25 model built. Call build_indexes() first.")
        
        
        if not hasattr(self.indexer, 'doc_ids'):
            raise AttributeError("Indexer does not have doc_ids attribute.")
        
        
        try:
            scores = self.indexer.bm25.get_scores(query_tok)
        except AttributeError as e:
            
            if hasattr(self.indexer.bm25, 'transform'):
                
                scores = self.indexer.bm25.transform(query_tok)
            else:
                raise e
        
        
        n_docs = len(scores)
        k = min(top_k, n_docs)
        top_idx = np.argsort(scores)[::-1][:k]
        
        
        results = [(self.indexer.doc_ids[i], float(scores[i])) 
                  for i in top_idx if scores[i] > 0]
        
        return results[:top_k] 
    
    def hybrid_search(self, query, top_k=5, tfidf_weight=0.5, bm25_weight=0.5):
        """
        Perform hybrid search combining TF-IDF and BM25 scores.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            tfidf_weight (float): Weight for TF-IDF scores (0 to 1)
            bm25_weight (float): Weight for BM25 scores (0 to 1)
            
        Returns:
            list: List of tuples (doc_id, score) for top_k documents
        """
        if tfidf_weight + bm25_weight != 1.0:
            raise ValueError("Weights must sum to 1.0")
        
        # Get individual results
        tfidf_results = dict(self.tfidf_search(query, top_k * 2))
        bm25_results = dict(self.bm25_search(query, top_k * 2))
        
        # Combine scores
        all_docs = set(list(tfidf_results.keys()) + list(bm25_results.keys()))
        combined_scores = {}
        
        for doc_id in all_docs:
            tfidf_score = tfidf_results.get(doc_id, 0)
            bm25_score = bm25_results.get(doc_id, 0)
            
            # Normalize scores if needed (optional)
            combined_score = (tfidf_weight * tfidf_score + 
                            bm25_weight * bm25_score)
            combined_scores[doc_id] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:top_k]
        
        return sorted_results


# Optional: Add a main function for testing
if __name__ == "__main__":
    # Simple test when run directly
    print("Testing RetrievalEngine class...")
    
    # Mock Indexer for testing
    class MockIndexer:
        def __init__(self):
            self.doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            # Mock TF-IDF attributes
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer()
            texts = ["hello world", "machine learning", "information retrieval"]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
         
            class MockBM25:
                def get_scores(self, query):
                    return np.random.rand(5)
            self.bm25 = MockBM25()
    
    try:
     
        mock_indexer = MockIndexer()
        engine = RetrievalEngine(mock_indexer)
        
     
        print("\nTesting TF-IDF search:")
        results = engine.tfidf_search("hello world", 3)
        print(f"Results: {results}")
        
        
        print("\nTesting BM25 search:")
        results = engine.bm25_search("machine learning", 3)
        print(f"Results: {results}")
        
      
        print("\nTesting hybrid search:")
        results = engine.hybrid_search("information", 3)
        print(f"Results: {results}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")