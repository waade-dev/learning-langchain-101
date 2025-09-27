from starter import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel




messages =[
    ("system", "you are an expert in {topic}"),
    ("human", "based on the specified topic give {count} number of features on on it")
]

pros_message =[
    ("system", "you are an expert in {topic}"),
    ("human", "based on the specified {features} tell me 3 pros of buying it in AI era")
]

cons_message =[
    ("system", "you are an expert in {topic}"),
    ("human", "based on the specified {features} tell me 3 cons of buying it in AI era")
]

message_template = ChatPromptTemplate.from_messages(messages= messages)
pros_template = ChatPromptTemplate.from_messages(messages= pros_message)
cons_template = ChatPromptTemplate.from_messages(messages= cons_message)


chain = (
    {
        "topic": (lambda x: x["topic"]),
        "features": (
                message_template|
                llm|
                StrOutputParser()
        )
    } |
    RunnableParallel({
        "pros": pros_template| llm | StrOutputParser(),
        "cons": cons_template| llm | StrOutputParser()
    }) |
    RunnableLambda( lambda x: f"pros: {x["pros"]} \n cons: {x["cons"]}")
)

print(chain.invoke({"topic": "macbook", "count": "3"}))
