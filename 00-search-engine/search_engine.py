import pandas as pd
import requests
import logging
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

# Download the document json from the github url

docs_url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"

try:
    response = requests.get(url=docs_url)
except Exception as e:
    logging.error(f"Failed to downloads docs due to: {e}")
else:
    docs_json = response.json()
    docs_df = pd.json_normalize(docs_json, "documents", ["course"])
    docs_df = docs_df[["course", "section", "question", "text"]]

print(docs_df.head())
print(docs_df[docs_df["course"] == "data-engineering-zoomcamp"].head())

cv = CountVectorizer(min_df=5)
cv.fit(docs_df["text"])
print(cv.get_feature_names_out())