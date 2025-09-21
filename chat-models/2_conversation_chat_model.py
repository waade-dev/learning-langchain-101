# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")



llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)


messages= [
    SystemMessage(content="you are a math prof"),
    HumanMessage(content="whats 81 by 9"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="whats by div 9 of that")
]

response = llm.invoke(messages)
print(response.content)