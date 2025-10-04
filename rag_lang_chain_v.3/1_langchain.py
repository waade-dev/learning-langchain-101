from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/amith.k/Developer/lang_chain_basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)

message = [
    ("system", "You are a helpful assistant that corrects the grammer and make it professional"),
    ("human", " hat are the mai comonents of an LLM-powered autonomous agent system, pls fix thos")
]
data = llm.invoke(message).content
print(data)