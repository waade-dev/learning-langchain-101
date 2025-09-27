from starter import llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


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
