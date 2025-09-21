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

conversations = []
conversations.append(SystemMessage(content="you are a genius in all the conversation pls clear all the users doubt"))
while(True):
    userInput = input("pls enter the prompt: \n")
    
    if userInput == "kill":
        break

    conversations.append(HumanMessage(content=userInput))

    ai_response = llm.invoke(conversations)
    print("\n"+ai_response.content+"\n")

    conversations.append(AIMessage(content=ai_response.content))
    

