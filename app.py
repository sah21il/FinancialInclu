from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.environ["PINECONE_API_KEY"] = "pcsk_49sxre_FaLMUy2b8ScAXFBkS6Mm1uaBuUGhn1kKXdPXdVxq4tixgmgGuQsKXQScuPxdTex"
HUGGINGFACEHUB_API_KEY=os.environ["HGGINGFACEHUB_API_KEY"] = "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"

embeddings = download_hugging_face_embeddings()

index_name = "eyes5"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = HuggingFaceHub(
    repo_id="gpt2", 
    huggingfacehub_api_token="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",
    model_kwargs={
        "temperature": 0.4,  # Adjust creativity
        "max_new_tokens": 500,  # Adjust token limit
    }
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chattem.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User message: {msg}")
    
    # Invoke the RAG (retrieval-augmented generation) chain
    response = rag_chain.invoke({"input": msg})
    
    # Output the response for debugging purposes
    print(f"Response: {response['answer']}")
    
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
