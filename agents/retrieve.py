import logging
import json
from langchain_community.embeddings import OllamaEmbeddings

class Retrieve:
    def __init__(self, collection, embeddings, prompttemplate):
        self.collection = collection
        self.embeddings = embeddings
        self.prompttemplate = prompttemplate

    def retrieve_relevant_json_rows(self, csv_row):
        query = json.dumps(csv_row.to_dict())
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(query_embeddings=[query_embedding], n_results=5)
        logging.info(f"Query results: {results}")

        if results and results['documents']:
            relevant_rows = [json.loads(doc) for doc in results['documents'][0]]
            return relevant_rows
        else:
            logging.warning("No relevant documents found in results.")
            return []

    def update(self, csv_row):
        relevant_json_rows = self.retrieve_relevant_json_rows(csv_row)

        input_data = {
            "csv_row": csv_row,
            "json_rows": relevant_json_rows
        }
    
        updated_row = self.prompttemplate.chain.invoke(input_data)
        return updated_row
