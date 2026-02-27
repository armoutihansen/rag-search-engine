from google import genai
from .constants import GEMINI_MODEL


def rag(query: str, results: list, api_key: str) -> None:
    """
    Perform Retrieval Augmented Generation (RAG) by taking a query and search results,
    and generating an answer using the retrieved information.

    Args:
        query (str): The search query.
        results (list): A list of retrieved documents relevant to the query.
        api_key (str): API key for the language model service.

    Returns:
        None: Prints the generated answer based on the query and retrieved documents.
    """
    client = genai.Client(api_key=api_key)
    context = "\n".join([f"{result['title']} - {result['document']}" for result in results])
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{context}

Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt)
    
    print("Search results:")
    for result in results:
        print(f"- {result['title']}")
    
    print("\nRAG Response:")
    print(response.text)

def summarize(query: str, results: list, api_key: str) -> None:
    """
    Summarize a long text document using the LLM, based on a query and retrieved results.

    Args:
        query (str): The search query.
        results (list): A list of retrieved documents relevant to the query.
        api_key (str): API key for the language model service.

    Returns:
        None: Prints the generated summary based on the query and retrieved documents.
    """
    client = genai.Client(api_key=api_key)
    context = "\n".join([f"{result['title']} - {result['document']}" for result in results])
    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{context}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt)
    print("Search results:")
    for result in results:
        print(f"- {result['title']}")
    print("\nLLM Summary:")
    print(response.text)
    
def citations(query: str, results: list, api_key: str) -> None:
    """
    Generate a summary with citations to the retrieved documents based on a query.

    Args:
        query (str): The search query.
        results (list): A list of retrieved documents relevant to the query.
        api_key (str): API key for the language model service.

    Returns:
        None: Prints the generated summary with citations based on the query and retrieved documents.
    """
    client = genai.Client(api_key=api_key)
    context = "\n".join([f"{result['title']} - {result['document']}" for result in results])
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{context}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt)
    print("Search results:")
    for result in results:
        print(f"- {result['title']}")
    print("\nLLM Answer:")
    print(response.text)
    
def question_answering(question: str, results: list, api_key: str) -> None:
    """
    Answer a question using RAG with retrieved documents.

    Args:
        question (str): The question to answer.
        results (list): A list of retrieved documents relevant to the question.
        api_key (str): API key for the language model service.

    Returns:
        None: Prints the generated answer based on the question and retrieved documents.
    """
    client = genai.Client(api_key=api_key)
    context = "\n".join([f"{result['title']} - {result['document']}" for result in results])
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt)
    print("Search results:")
    for result in results:
        print(f"- {result['title']}")
    print("\nAnswer:")
    print(response.text)