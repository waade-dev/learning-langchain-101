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

import json
docs = []
with open("/Users/amith.k/Developer/lang_chain_basics/rag_lang_chain_v.3/sample.json") as f:
    docs = json.load(f)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chunks = []
for doc in docs:
    chunks.append({
        "id": doc["id"],
        "text": f"{doc['title']}\n{doc['content']}"
    })
    print(chunks)


docs_embeddings = embeddings.embed_documents([chunk["text"] for chunk in chunks])
texts = [chunk['text'] for chunk in chunks]

# print(docs_embeddings[0])

user_query = input("enter your question \n ")
query_embedding = embeddings.embed_query(user_query)

# similarity = cosine_similarity([query_embedding], docs_embeddings)
# print(similarity.argmax(axis=1)[0])
# print(chunks[similarity.argmax(axis=1)[0]])


db = FAISS.from_texts(texts=texts, embedding=embeddings)
print(db.similarity_search(user_query))

db.save_local("/Users/amith.k/Developer/lang_chain_basics/rag_lang_chain_v.3/faiss_index")

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vectorstore = FAISS.load_local("/Users/amith.k/Developer/lang_chain_basics/rag_lang_chain_v.3/faiss_index", embeddings, allow_dangerous_deserialization=True)
# print(vectorstore)
# user_query = input("enter your question \n ")
# print(vectorstore.similarity_search(user_query, k=3))