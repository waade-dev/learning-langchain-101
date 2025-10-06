from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


import json
docs = []
load_dotenv("/Users/amith.k/Developer/lang_chain_basics/.env")
with open("/Users/amith.k/Developer/lang_chain_basics/rag_lang_chain_v.3/sample.json") as f:
    docs = json.load(f)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)

print(docs)
memory=[]

chunks=[]
for doc in docs:
    chunks.append({
        "id": doc["id"],
        "text": f"{doc['title']}\n{doc['content']}"
    })

print(chunks)
# db = FAISS.from_texts(texts=[chunk["text"] for chunk in chunks], embedding=embeddings)
db = FAISS.load_local("/Users/amith.k/Developer/lang_chain_basics/rag_lang_chain_v.3/faiss_index", embeddings, allow_dangerous_deserialization=True)

def rag_chain(data):
    query = data["input"]
    similar = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in similar])
    return {"query": query, "context": context, "history": data["history"]}




inital_prompt = [
    SystemMessage("You are an English grammar correction assistant. Your task is to only correct the grammar and spelling of the user's text, without adding any explanations or extra information.")
]

while(True):
    input_promt = input("enter the prompt for rag \n")
    tempMessage = inital_prompt+ [HumanMessage(input_promt)]
    clean_input = llm.invoke(tempMessage)
    print(clean_input.content)
    memory.append(HumanMessage(clean_input.content))
    chain = (
        RunnableLambda(lambda x: {"input": clean_input.content, "history": memory} )
        | RunnableLambda(rag_chain)
        | RunnableLambda(lambda d: f"History:\n" +
                   "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage)
                              else f"AI: {m.content}" for m in d["history"]]) +
                   f"\n\nContext:\n{d['context']}\n\nQuestion: {d['query']}")

        |llm 
        |StrOutputParser())
    
    # final_prompt = (
    #     "History:\n" +
    # "\n".join([f"User: {memory}" if isinstance(m, HumanMessage) else f"AI: {memory}" 
    #            for m in memory]) +
    # f"\n\nContext:\n{memory}\n\nQuestion: {memory}"
    # )


    # print("\n--- Prompt to LLM ---\n", final_prompt)
    ai_message = chain.invoke(None)
    memory.append(AIMessage(ai_message))