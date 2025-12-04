# indexer.py
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from preprocess import preprocess  

class Indexer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.documents = []       
        self.doc_ids = []         
        self.corpus_tokens = []   
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None

    def load_docs(self):
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path not found: {self.dataset_path}")

        for fname in sorted(os.listdir(self.dataset_path)):
            fpath = os.path.join(self.dataset_path, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(fpath, "r", encoding="latin-1") as f:
                        text = f.read()

                tokens = preprocess(text)
                if len(tokens) == 0:
                    
                    tokens = []

                self.documents.append(text)
                self.doc_ids.append(fname)
                self.corpus_tokens.append(" ".join(tokens))

        if len(self.documents) == 0:
            raise ValueError(f"No documents loaded from {self.dataset_path}")

    def build_tfidf(self):
        
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_tokens)

    def build_bm25(self):
        corpus = [doc.split() for doc in self.corpus_tokens]
        self.bm25 = BM25Okapi(corpus)

    def build_indexes(self):
        print("Loading documents...")
        self.load_docs()

        print(f"Loaded {len(self.documents)} documents.")

        print("Building TF-IDF index...")
        self.build_tfidf()

        print("Building BM25 index...")
        self.build_bm25()

        print("Indexing complete.")
