# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


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

case_handelling = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"count: {len(x)} \n{x}")

chain = message_template | llm | StrOutputParser()  | case_handelling| count_words

print(chain.invoke({"topic": "noob in coc", "count": "3"}))