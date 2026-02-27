from google import genai
from google.genai import types

from .constants import GEMINI_MODEL


def describe_image(img: bytes, mime: str, query: str, api_key: str) -> None:
    """Describe an image using a language model.

    Args:
        img: The image data in bytes.
        mime: MIME type of the image.
        query: The text query to improve based on the image content.
        api_key: API key for the language model service.

    Returns:
        None: Prints the generated description of the image.
    """
    client = genai.Client(api_key=api_key)
    prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    
    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip()
    ]
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts)
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    