# Import Azure OpenAI
from starter import llm
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


message = [
    ("system", "you are a {topic} comedian"),
    ("human", "tell me {count} joke about it")
]

message_template = ChatPromptTemplate.from_messages(messages=message)

case_handelling = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"count: {len(x)} \n{x}")

chain = message_template | llm | StrOutputParser()  | case_handelling| count_words

print(chain.invoke({"topic": "noob in coc", "count": "3"}))