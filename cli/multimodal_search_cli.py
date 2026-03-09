import argparse

from dotenv import load_dotenv
from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding generation"
    )
    verify_parser.add_argument(
        "image_path", type=str, help="Path to the image to verify embedding generation"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search for similar images using an input image query"
    )
    image_search_parser.add_argument(
        "image_path", type=str, help="Path to the image to use as a search query"
    )
    image_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image_path
            verify_image_embedding(image_path)
        case "image_search":
            image_path = args.image_path
            limit = args.limit
            image_search_command(image_path, limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
