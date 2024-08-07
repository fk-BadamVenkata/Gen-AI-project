import pandas as pd
class CustomOpenAIEmbeddingFunction:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
    
    def __call__(self, input):
        text = input['input'] 
        embedding = self.embeddings_model.embed_query(text)
        if isinstance(embedding, list):
            return embedding  
        else:
            return embedding.tolist()  
csv_file_path = 'table2.csv'
csv_data = pd.read_csv(csv_file_path)