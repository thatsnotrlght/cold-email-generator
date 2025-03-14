from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd

# Initializes LLM with specified parameters
llm = ChatGroq(
    temperature = 0,
    groq_api_key = "api_key",
    model_name = "llama-3.3-70b-versatile"
)

# loads web page content
loader = WebBaseLoader("https://maximus.avature.net/careers/FolderDetail/United-States-Intern-AI-Software-Engineering/28377")
page_data = loader.load().pop().page_content

# defines prompt template for extracting job posting details
prompt_extract = PromptTemplate.from_template(
    """
    ### SCRAPED TEXT FROM WEBSITE:
    {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job posting and return them in JSON format containing following keys: 'role', 'experience', 'skills', and 'description'.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):
"""
)

# Pipe operator to chain the response from the model to the prompt
chain_extract = prompt_extract | llm
chainResponse = chain_extract.invoke(input={"page_data": page_data})
# print("\nString format response")
# print(chainResponse.content)


json_parser = JsonOutputParser()
json_response = json_parser.parse(chainResponse.content)
# print("\nJSON format response")
# print(json_response)

job = json_response
print(job['skills'])