from collections import Counter, defaultdict
import math
import string
from nltk.stem import PorterStemmer
from .constants import BM25_K1, BM25_B
import os
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.stemmer = PorterStemmer()
        self.doc_lengths = defaultdict(int)
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def _preprocess(self, text: str) -> str:
        """Remove punctuation and convert to lowercase.

        Args:
            text (str): _text to preprocess_

        Returns:
            str: _preprocessed text_
        """
        translator = str.maketrans("", "", string.punctuation)
        return text.lower().translate(translator)

    def _tokenize(self, text: str) -> list:
        """Tokenize preprocessed text and stem each token.

        Args:
            text (str): _preprocessed text to tokenize_

        Returns:
            list: _list of stemmed tokens_
        """
        with open("./data/stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set(f.read().splitlines())
        tokens = text.split()
        return [self.stemmer.stem(token) for token in tokens if len(token) > 0 and token not in stopwords]

    def __add_document(self, doc_id: int, text: str) -> None:
        """Tokenize the input text, then add each token to the index with the document ID.

        Args:
            doc_id (int): _document id_
            text (str): _document text_
        """
        text = self._preprocess(text)
        tokens = self._tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[token].update([doc_id])
        self.doc_lengths[doc_id] += len(tokens)
            
    def get_documents(self, term: str) -> list:
        """Return the set of document IDs that contain the given term.

        Args:
            term (str): _term to search for_

        Returns:
            list: _list of document IDs containing the term_
        """
        return list(sorted(self.index.get(term.lower(), set())))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        text = self._preprocess(term)
        tokens = self._tokenize(text)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        return self.term_frequencies[tokens[0]][doc_id]
    
    def get_idf(self, term: str) -> float:
        text = self._preprocess(term)
        tokens = self._tokenize(text)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        num_docs_with_term = len(self.index.get(term, set()))
        total_docs = len(self.docmap)
        return math.log((total_docs + 1) / (num_docs_with_term + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        text = self._preprocess(term)
        tokens = self._tokenize(text)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        term = tokens[0]
        num_docs_with_term = len(self.index.get(term, set()))
        total_docs = len(self.docmap)
        return math.log((total_docs - num_docs_with_term + 0.5) / (num_docs_with_term + 0.5) + 1)
    
    
    
    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def build(self, movies):
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie.get('description', '')}")
    
    def save(self) -> None:
        """save the index and docmap attributes to disk using the pickle module's dump function.
. Save as two separate files: cache/index.pkl and cache/docmap.pkl. Will create cache directory if it does not exist.

        """
        
        os.makedirs("cache", exist_ok=True)
        
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
            
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
            
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)
            
    def load(self) -> None:
        """load the index and docmap attributes from disk using the pickle module's load function. Load from two separate files: cache/index.pkl and cache/docmap.pkl.

        """
        import pickle
        
        try: 
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
            
            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Index files not found. Please run the 'build' command to create the index before loading.")
        
    def bm25_idf_command(self, term: str) -> float:
        idx = InvertedIndex()
        try:
            idx.load()
        except FileNotFoundError:
            print("Index files not found. Please run the 'build' command to create the index before searching.")
            raise FileNotFoundError("Index files not found. Please run the 'build' command to create the index before loading.")
        return idx.get_bm25_idf(term)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = (1 - b) + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) if tf > 0 else 0
    
    def bm25_tf_command(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        idx = InvertedIndex()
        try:
            idx.load()
        except FileNotFoundError:
            print("Index files not found. Please run the 'build' command to create the index before searching.")
            raise FileNotFoundError("Index files not found. Please run the 'build' command to create the index before loading.")
        return idx.get_bm25_tf(doc_id, term, k1, b)
    
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.docmap)
        return total_length / num_docs if num_docs > 0 else 0
    
    def bm25(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term, k1, b)
        return idf * tf
    
    def bm25_search(self, query: str, limit: int,k1: float = BM25_K1, b: float = BM25_B) -> list:
        query = self._preprocess(query)
        tokens = self._tokenize(query)
        scores = defaultdict(float)
        
        for doc_id in self.docmap:
            score = 0.0
            for token in tokens:
                score += self.bm25(doc_id, token, k1, b)
            scores[doc_id] = score
            
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "document": self.docmap[doc_id].get("description", ""),
                "score": score
            }
            for doc_id, score in sorted_docs[:limit]
        ]