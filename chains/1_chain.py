# Import Azure OpenAI
from starter import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "you are a comedian who tell joke about {topic}"),
    ("human", "Tell me {count} jokes on it")
])

chain =   prompt_template |llm |  StrOutputParser()

result = chain.invoke({"topic": "lawyers", "count": "3"})
print(result)