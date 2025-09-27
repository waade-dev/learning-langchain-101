from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel, RunnableBranch


load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)


messages =[
    ("system", "you are an expert in {topic}"),
    ("human", "based on the specified topic give it a response if the product is good ot bad,i want either good or bad not both")
]

pros_message = [
    ("system", "you are an expert in {topic}"),
    ("human", "tell me why is it good")
]

cons_message = [
    ("system", "you are an expert in {topic}"),
    ("human", "tell me why is it bad")
]


message_template = ChatPromptTemplate.from_messages(messages= messages)
pros_template = ChatPromptTemplate.from_messages(messages= pros_message)
cons_template = ChatPromptTemplate.from_messages(messages= cons_message)



chain = (
    {
        "topic": RunnableLambda(lambda x: x["topic"]),
        "status": message_template | llm | StrOutputParser()
    } |
    RunnableBranch(
        
            (
                lambda x: "good" in str(x["status"]).lower(),
                RunnableLambda(lambda x: {"topic": x["topic"]}) | pros_template | llm | StrOutputParser()
            ),
            (
                lambda x: "bad" in str(x["status"]).lower(),
                RunnableLambda(lambda x: {"topic": x["topic"]}) | cons_template | llm | StrOutputParser()
            ),
            lambda x: x
        #    RunnableLambda(lambda x: {"topic": x["topic"]}) | pros_template | llm | StrOutputParser()
        
    ) |
    StrOutputParser()
)



print(chain.invoke({"topic": "macbook pro-2 2022"}))