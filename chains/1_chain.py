# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",   # must match your Azure deployment name
    temperature=0,
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "you are a comedian who tell joke about {topic}"),
    ("human", "Tell me {count} jokes on it")
])

chain =   prompt_template |llm |  StrOutputParser()

result = chain.invoke({"topic": "lawyers", "count": "3"})
print(result)