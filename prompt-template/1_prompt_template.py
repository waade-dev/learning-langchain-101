from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

#load from dot env
load_dotenv(dotenv_path="/Users/amith.k/Developer/lang-chain-basics/.env")


llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)

# for basic templates

# template = "Tell me joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template=template)

# response = llm.invoke(prompt_template.invoke({"topic": "cats"}))
# print(response.content)


#for messages
#Human message ("content=") doesnt work in this case

message = [
    ("system", 
     "You are a stand-up comedian. tell jokes in Kannada. "
     "The jokes must be about {topic}."),
    ("human", "Tell me {count} jokes.")
]

message_template = ChatPromptTemplate.from_messages(messages=message)

print(llm.invoke(message_template.invoke({"topic": "dad joke", "count": 3})).content)
