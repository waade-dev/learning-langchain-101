# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI
from starter import llm


# Invoke
response = llm.invoke("Tell me a joke")
print(response.content)

