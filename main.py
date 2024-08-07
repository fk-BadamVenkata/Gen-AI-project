import pandas as pd
import json
import logging
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from chromadb.config import Settings
from typing import Dict, Any
from agents.custom_openai_embedding_function import CustomOpenAIEmbeddingFunction
from agents.prompt_template import prompttemplate
from agents.retrieve import Retrieve

logging.basicConfig(level=logging.INFO)

csv_file_path = 'table2.csv'
csv_data = pd.read_csv(csv_file_path)

json_file_path = 'tabledata.json'
with open(json_file_path, 'r') as f:
    json_data = [json.loads(line) for line in f]
json_df = pd.DataFrame(json_data)

llm = Ollama(model="llama3")

embeddings = OllamaEmbeddings(model="llama3")

chroma_client = chromadb.Client(Settings())

embedding_function = CustomOpenAIEmbeddingFunction(embeddings)


collection = chroma_client.create_collection(
    name="json_collection",
    embedding_function=embedding_function
)

text_splitter = CharacterTextSplitter()

texts = json_df.apply(lambda row: json.dumps(row.to_dict()), axis=1).tolist()
split_texts = [text_splitter.split_text(text) for text in texts]
flattened_texts = [item for sublist in split_texts for item in sublist]

for idx, (text, flattened_text) in enumerate(zip(texts, flattened_texts)):
    embedding = embedding_function({'input': flattened_text})
    collection.add(
        ids=[str(idx)],
        documents=[text],
        embeddings=[embedding]
    )

retrieve_instance = Retrieve(collection, embeddings, prompttemplate)

updated_csv_data_rag = csv_data.apply(lambda row: retrieve_instance.update(row), axis=1)
updated_csv_data_rag = pd.DataFrame(list(updated_csv_data_rag))

updated_csv_file_path_rag = 'updated_table_rag.csv'
updated_csv_data_rag.to_csv(updated_csv_file_path_rag, index=False)

print(f"Updated CSV data with RAG saved to {updated_csv_file_path_rag}")
