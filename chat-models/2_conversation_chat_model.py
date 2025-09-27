# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
from starter import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage



messages= [
    SystemMessage(content="you are a math prof"),
    HumanMessage(content="whats 81 by 9"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="whats by div 9 of that")
]

response = llm.invoke(messages)
print(response.content)