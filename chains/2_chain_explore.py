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


message = [
    ("system", "you are a {topic} comedian"),
    ("human", "tell me {count} joke about it")
]
message_template = ChatPromptTemplate.from_messages(messages=message)

chain = message_template | llm | StrOutputParser()
print(chain.invoke({"topic": "america", "count": 3}))