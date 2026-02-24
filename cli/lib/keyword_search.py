from collections import Counter, defaultdict
import string
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.stemmer = PorterStemmer()

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
        tokens = text.split()
        return [self.stemmer.stem(token) for token in tokens if len(token) > 0]

    def _add_document(self, doc_id: int, text: str) -> None:
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
    
    def build(self, movies):
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self._add_document(doc_id, f"{movie['title']} {movie.get('description', '')}")
    
    def save(self) -> None:
        """save the index and docmap attributes to disk using the pickle module's dump function.
. Save as two separate files: cache/index.pkl and cache/docmap.pkl. Will create cache directory if it does not exist.

        """
        import os
        import pickle
        
        os.makedirs("cache", exist_ok=True)
        
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
            
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
            
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
        except FileNotFoundError:
            raise FileNotFoundError("Index files not found. Please run the 'build' command to create the index before loading.")