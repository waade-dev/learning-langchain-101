# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")



llm = AzureChatOpenAI(
    deployment_name="gpt-4o",   # must match your Azure deployment name
    temperature=0,
)

# Invoke
response = llm.invoke("Tell me a joke")
print(response.content)

