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

embeggings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

load_dotenv("/Users/amith.k/Developer/lang_chain_basics/.env")
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    temperature=0.4,
    top_p=0.9
)

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

document_template = [physics_template, math_template]
doc_embeddings = embeggings.embed_documents(document_template)


def similarity_search_function(input_question):
    query_embeddings = embeggings.embed_query(input_question)
    similarity = cosine_similarity([query_embeddings], doc_embeddings)
    print(similarity)
    print(similarity.argmax())
    most_favored = document_template[similarity.argmax(axis=1)[0]]
    fav = ChatPromptTemplate.from_template(most_favored)
    return fav



chain = ( RunnableLambda(similarity_search_function) | llm | StrOutputParser())



input_question = input("what is it you like to ask \n")
print(chain.invoke(input_question))

