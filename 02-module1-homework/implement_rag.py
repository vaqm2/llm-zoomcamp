#!/usr/bin/env python3

"Homework: Implement RAG (Retrieval-Augmented Generation) with Python & Elasticsearch"

import os
import logging
import requests
import pandas as pd
import elasticsearch as es
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option("display.max_colwidth", None)  # Show full text in DataFrame
pd.set_option("display.max_columns", None)  # Show all columns in DataFrame


def fetch_data(url) -> pd.DataFrame:
    """
    Fetch data from a given URL and return it as a pandas DataFrame.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses
    except requests.RequestException as e:
        logger.error("Error fetching data from %s: %s", url, e)
        return None
    return pd.json_normalize(response.json(), record_path="documents", meta=["course"])


def create_index(es_client, index_name: str, index_settings: dict) -> bool:
    """
    Create an Elasticsearch index if it does not already exist.

    Args:
        es_client (elasticsearch.Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the index to create.
        index_settings (dict): Settings and mappings for the index.
    """
    if es_client.indices.exists(index=index_name):
        logger.info("Index '%s' already exists. Skipping creation.", index_name)
        return False
    try:
        es_client.indices.create(index=index_name, body=index_settings)
    except Exception as e:
        logger.error("Error creating index '%s': %s", index_name, e)
        return False
    logger.info("Index '%s' created successfully.", index_name)
    return True


def index_data(es_client, index_name: str, data: pd.DataFrame) -> None:
    """
    Index data into the specified Elasticsearch index.

    Args:
        es_client (elasticsearch.Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the index to index data into.
        data (pd.DataFrame): DataFrame containing the data to index.
    """
    for _, row in data.iterrows():
        doc = {
            "course": row["course"],
            "question": row["question"],
            "section": row["section"],
            "text": row["text"],
        }
        es_client.index(index=index_name, document=doc)
    logger.info("Data indexed successfully into '%s'.", index_name)


def search(es_client, index_name: str, query: str) -> pd.DataFrame:
    """
    Search for documents in the specified Elasticsearch index.

    Args:
        es_client (elasticsearch.Elasticsearch): Elasticsearch client instance.
        index_name (str): Name of the index to search.
        query (str): Search query string.

    Returns:
        pd.DataFrame: DataFrame containing the search results.
    """
    results = []
    search_body = {
        "size": 5,  # Limit the number of results
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields",
                    }
                },
            }
        },
    }
    response = es_client.search(index=index_name, body=search_body)
    for hit in response["hits"]["hits"]:
        results.append(
            {"text": hit["_source"]["text"]},
        )
    if results:
        return pd.DataFrame(results)
    return None


def build_prompt_for_gemini(question: str, context: pd.DataFrame) -> str:
    """
    Build a prompt for the LLM using the question and context.

    Args:
        question (str): The user's question.
        context (pd.DataFrame): DataFrame containing the context for the question.

    Returns:
        str: Formatted prompt string.
    """
    prompt = """
    You are a course assistant for the Datatalks Club.  
    Answer the question based on the retrieved context.
    If the answer is not in the context, say you don't have enough information.
    """
    prompt += f"\n\nQuestion: {question}\n\n"
    prompt += "Context:\n\n"
    for _, row in context.iterrows():
        prompt += f"{row['text']}\n\n"
    return prompt.strip()


def make_api_call_to_gemini(prompt: str) -> str:
    """
    Make an API call to Google Gemini with the provided prompt.

    Args:
        prompt (str): The prompt to send to the Gemini model.

    Returns:
        str: The response text from the Gemini model.
    """
    load_dotenv()  # Load environment variables from .env file
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set.")
        return "Error: API key not found."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        logger.error("Error calling Gemini API: %s", e)
        return "Error: Failed to get response from Gemini."
    return response.text


def main() -> None:
    """
    Main function to execute the RAG implementation.
    """
    data_df = fetch_data(
        "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1"
    )
    if data_df is None:
        logger.error("Failed to fetch data. Exiting.")
        return
    logger.info("Data fetched successfully, %d records found.", len(data_df))
    logger.info("Creating Elasticsearch index and indexing data...")
    es_client = es.Elasticsearch("http://localhost:9200")
    index_name = "rag_index"
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "course": {"type": "keyword"},
                "question": {"type": "text"},
                "section": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }
    if es_client.indices.exists(index=index_name):
        logger.info("Index already exists or failed to create. Skipping indexing.")
    else:
        index_created = es_client.indices.create(index=index_name, body=index_settings)
        if index_created:
            index_data(es_client, index_name, data_df)
            logger.info("Data indexed successfully.")
    queries = [
        "How do I copy a file to a docker container?",
        "How do I execute a command in a kubernetes pod?",
        "How many zoomcamp courses are there?",
    ]
    for query in queries:
        results = search(es_client, index_name, query)
        if results is not None and not results.empty:
            prompt = build_prompt_for_gemini(query, results)
            response = make_api_call_to_gemini(prompt)
            print(f"Question: {query}")
            print(f"Response: {response}")


if __name__ == "__main__":
    main()
