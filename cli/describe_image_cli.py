import argparse
import os
import mimetypes

from dotenv import load_dotenv
from lib.describe_image import describe_image


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--query", type=str, help="Query to describe the image")
    parser.add_argument("--image", type=str, help="Path to the image to describe")
    args = parser.parse_args()
    query = args.query
    image_path = args.image
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"  # Default to jpeg if unknown
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    describe_image(image_data, mime, query, api_key)
    
if __name__ == "__main__":
    main()