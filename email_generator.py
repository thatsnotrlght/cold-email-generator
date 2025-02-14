from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)

#response = llm.invoke("What is this project going to entail? ")
# print(response.content)

loader = WebBaseLoader("https://maximus.avature.net/careers/FolderDetail/United-States-Intern-IT-Software-Engineering/27568")
page_data = loader.load().pop().page_content

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
print(chainResponse.content)