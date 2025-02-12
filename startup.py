from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)

response = llm.invoke("What is this project going to entail? ")
print(response.content)