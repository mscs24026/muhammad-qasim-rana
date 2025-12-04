import time

def precision_at_k(results, relevant_docs, k=5):
    retrieved = set([doc for doc, _ in results[:k]])
    relevant = set(relevant_docs)
    return len(retrieved & relevant) / k

def evaluate_query(engine, query, relevant_docs, k=5):
    start = time.time()
    tfidf_res = engine.tfidf_search(query, k)
    bm25_res = engine.bm25_search(query, k)
    end = time.time()

    return {
        "TF-IDF_P@K": precision_at_k(tfidf_res, relevant_docs, k),
        "BM25_P@K": precision_at_k(bm25_res, relevant_docs, k),
        "Response_Time_sec": end - start
    }
