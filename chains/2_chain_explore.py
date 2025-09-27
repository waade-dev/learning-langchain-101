# Import Azure OpenAI
from starter import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


message = [
    ("system", "you are a {topic} comedian"),
    ("human", "tell me {count} joke about it")
]
message_template = ChatPromptTemplate.from_messages(messages=message)

chain = message_template | llm | StrOutputParser()
print(chain.invoke({"topic": "america", "count": 3}))