from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="path to .env")
# should contain AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION
#ps: you can use any other llm

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)
