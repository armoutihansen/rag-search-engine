import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .constants import MULTIMODAL_MODEL
from .semantic_search import cosine_similarity
from .utils import load_movies_data

class MultimodalSearch:
    def __init__(self, documents: list, model_name: str = MULTIMODAL_MODEL):
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc.get('description', '')}" for doc in documents]
        self.model = SentenceTransformer(model_name)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def generate_image_embedding(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_embedding = self.model.encode([image])
        return image_embedding[0]
    
    def search_with_image(self, image_path: str, limit: int = 5):
        image_embedding = self.generate_image_embedding(image_path)
        similarities = np.array(
            [
                cosine_similarity(image_embedding, text_embedding)
                for text_embedding in self.text_embeddings
            ]
        )
        
        # Sort by similarity score in descending order
        sorted_indices = np.argsort(similarities)[::-1][:limit]
        
        # Return list of dicts with document info and similarity score
        return [
            {
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "description": self.documents[idx].get("description", ""),
                "score": float(similarities[idx])
            }
            for idx in sorted_indices
        ]
        

def verify_image_embedding(image_path: str) -> None:
    """Verify that image embedding generation works.
    
    Args:
        image_path: Path to the image file
    """
    documents = load_movies_data()
    search = MultimodalSearch(documents)
    image_embedding = search.generate_image_embedding(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")
    
def image_search_command(image_path: str, limit: int = 5) -> None:
    """Search for movies using an image query.
    
    Args:
        image_path: Path to the image file
        limit: Number of results to return
    """
    documents = load_movies_data()
    search = MultimodalSearch(documents)
    results = search.search_with_image(image_path, limit)
    print(f"Search results for image: '{image_path}'")
    for i, result in enumerate(results, start=1):
        print(
            f"{i}. {result['title']} (similarity: {result['score']:.3f})\n  {result['description'][:200]}"
        )