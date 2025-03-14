import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio


def create_streamlit_app(llm, portfolio):
    st.title("Cold Mail Generator")

    # input field for user URL input
    url_input = st.text_input("Enter a URL:", value="https://beigene.wd5.myworkdayjobs.com/en-US/BeiGene/job/Summer-Internship--Large-Language-Model--LLM--Data-Scientist-AI-Engineer-Intern_R28023")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            # loads web page content
            loader = WebBaseLoader([url_input])
            data = loader.load().pop().page_content

            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            for job in jobs:
                skills = job.get("skills", [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # initilize the Chain and Portfolio objects
    chain = Chain()
    portfolio = Portfolio()

    st.set_page_config(layout="wide", page_title="Cold Mail Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio)