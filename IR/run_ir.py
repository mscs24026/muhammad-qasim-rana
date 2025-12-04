from indexer import Indexer
from retrieval import RetrievalEngine
from evaluate import evaluate_query

def main():
    dataset_path = "Articles.csv"  
     
    print("Building indexes...")
    indexer = Indexer(dataset_path)
    indexer.build_indexes() 
    engine = RetrievalEngine(indexer)
    
    print("\nSimple local IR system (TF-IDF + BM25). Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your search query (or 'exit'): ").strip()
        
        if query.lower() == "exit":
            print("Exiting.")
            break
        
        if not query:
            print("Please enter a non-empty query.")
            continue
         
        print("\n--- TF-IDF Results ---")
        tfidf_results = engine.tfidf_search(query, 5)
        if tfidf_results:
            for i, (doc, score) in enumerate(tfidf_results, 1):
                print(f"{i}. {doc} | Score: {score:.6f}")
        else:
            print("No results found.")
         
        print("\n--- BM25 Results ---")
        bm25_results = engine.bm25_search(query, 5)
        if bm25_results:
            for i, (doc, score) in enumerate(bm25_results, 1):
                print(f"{i}. {doc} | Score: {score:.6f}")
        else:
            print("No results found.") 
        print("\n--- Evaluation ---")
        print("To evaluate this query, enter document IDs that are relevant (comma-separated).")
        print("Press Enter to skip evaluation.")
        
        relevant_input = input("Relevant document IDs: ").strip()
        
        if relevant_input:
            try:
                relevant_docs = [doc_id.strip() for doc_id in relevant_input.split(',')]

                print(f"Evaluation would run with relevant docs: {relevant_docs}")

            except Exception as e:
                print(f"Error in evaluation input: {e}")
        else:
            print("Evaluation skipped.")

if __name__ == "__main__":
    main()