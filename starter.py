from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/amith.k/Developer/lang_chain_basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)