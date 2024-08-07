from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
class prompttemplate:
    prompt_template = PromptTemplate(
    input_variables=["csv_row", "json_rows"],
    template="""
    You are an AI assistant. Your task is to update missing fields in the CSV row using relevant information from the JSON data.
    Additionally, correct any misspelled words in the CSV row based on the context provided by the JSON data. Ensure all fields are accurate and correct.

    CSV Row:
    {csv_row}
    
    JSON Rows:
    {json_rows}
    
    Updated and Corrected CSV Row:
    """
    )

    llm = Ollama(model="llama3")
    chain = prompt_template | llm
    
    
