# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel


load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",   # must match your Azure deployment name
    temperature=0,
)


messages = [
    ("system", "you are an expert in {topic}"),
    ("human", "generate {count} of features of this")
]

pros_message = [
    ("system", "you are an expert on {topic}"),
    ("human", "given the features {features} tell me pros of buying this ")
]

cons_message = [
    ("system", "you are an expert on {topic}"),
    ("human", "given the features {features} tell me cons of buying this ")
]


message_template = ChatPromptTemplate.from_messages(messages=messages)
pros_template = ChatPromptTemplate.from_messages(messages=pros_message)
cons_template = ChatPromptTemplate.from_messages(messages=cons_message)


pros_chain = pros_template | llm | StrOutputParser()
cons_chain = cons_template | llm | StrOutputParser()


def combine_pros_cons(x, y):
    return f"pros: {x}\n \n cons: {y}"

chain = ({
    "topic": lambda x: x["topic"], 
    "features": (
        message_template |
        llm | 
        StrOutputParser() 
    )}|
    
    RunnableParallel({
        "pros": (lambda x: {"topic": x["topic"], "features": x["features"]}) | pros_chain,
        "cons": (lambda x: {"topic": x["topic"], "features": x["features"]}) | cons_chain,
    })
    |    RunnableLambda(lambda x: combine_pros_cons(x["pros"], x["cons"]))

)

print(chain.invoke({"topic": "macbook", "count": "3"}))