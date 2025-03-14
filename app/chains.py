import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Looks for .env file and sets personal API key as environemnt variable
load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
        temperature = 0,
        groq_api_key = os.getenv("GROQ_API_KEY"),
        model_name = "llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        extractedPrompt = PromptTemplate.from_template(
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

        chain_extract = extractedPrompt | self.llm
        result = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            result = json_parser.parse(result.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse through jobs.")
        return result if isinstance(result, list) else [result]
    
    def write_mail(self, job, links):
        # define prompt template for writing an email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            {job}
            ### INSTRUCTION:
            Write an email to the hiring manager expressing your interest in the job.
            Include the following in the email:
            - Why you are interested in the job
            - Your experience and skills that make you a good fit for the job
            - A call to action to schedule an interview
            From the job mentioned above, describe the capability of how I can fulfull their needs and add the most relevant ones from the following links to showcase the portfolios: {links}
            Do not provide a preamble.
            ### EMAIL TEMPLATE (NO PREAMBLE):

            """
        )

        # chains the response from model to prompt
        chain_extract = prompt_email | self.llm
        result = chain_extract.invoke({"job": str(job), "links": links})
        return result.content
        
     
#if __name__ == "__main__":
#    print(os.getenv("GROQ_API_KEY"))