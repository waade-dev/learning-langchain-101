from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


load_dotenv(dotenv_path="/Users/amith.k/Developer/lang_chain_basics/.env")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0,
)

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

print(vector_store)
print(index)