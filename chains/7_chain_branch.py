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
cons_message=[
    ("system", "you are an expert in {topic}"),
    ("human", "tell me 10 bad things on this topic")
]
pros_message=[
    ("system", "you are an expert in {topic}"),
    ("human", "tell me 10 good things on this topic")
]

message_template = ChatPromptTemplate.from_messages(messages= messages)
cons_template = ChatPromptTemplate.from_messages(messages= cons_message)
pros_template = ChatPromptTemplate.from_messages(messages= pros_message)


chain = (
    message_template|
    llm|
    StrOutputParser()
)

chain=(
    {
        "topic": lambda x: x["topic"],
        "status": message_template|
                    llm|
                    StrOutputParser()
    }|

    RunnableBranch(
        (lambda x: "good" in str(x["status"]).lower(), pros_template| llm| StrOutputParser()),
        (lambda x: "bad" in str(x["status"]).lower(), cons_template| llm| StrOutputParser()),
        lambda x: x
    )

)

print(chain.invoke({"topic": "1080Ti"}))


