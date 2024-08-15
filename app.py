import json
import os
import sys
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

## Data Ingestion

def data_ingestion(pdf):
    with open("temp_pdf.pdf", "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader("temp_pdf.pdf")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store

def get_vector_store(docs, index_name):
    if not docs:
        raise ValueError("The document list is empty. Please check the data ingestion process.")
    
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local(index_name)

def get_claude_llm():
    ## create the Anthropic Model
    llm=Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0", 
        client=bedrock,
        model_kwargs={'max_gen_len':512}
    )
    return llm

def get_llama2_llm():
    ## create the Anthropic Model
    llm=Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0", 
        client=bedrock,
        model_kwargs={'max_gen_len':512}
    )
    return llm

prompt_template = """
Human: You are provided with context from two different documents. 
Use the information from both documents to answer the question at the end. 
If the documents provide conflicting information, highlight the differences. 
Summarize with at least 250 words with detailed explanations. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

<Document 1 Context>
{context1}
</Document 1 Context>

<Document 2 Context>
{context2}
</Document 2 Context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context1", "context2", "question"]
)



def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query":query})
    return answer['result']


def main():
    st.set_page_config(page_title="Compare and Contrast two PDFs", page_icon=":books:")
    st.header("Chat and compare two PDFs using AWS Bedrock")

    user_question = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Upload two PDF files.")
        pdf1 = st.file_uploader("Upload the first document", type="pdf")
        pdf2 = st.file_uploader("Upload the second document", type="pdf")

        if st.button("Confirm Upload"):
            with st.spinner("Processing..."):
                doc1 = data_ingestion(pdf1)
                doc2 = data_ingestion(pdf2)

                get_vector_store(doc1, "faiss_index1")
                get_vector_store(doc2, "faiss_index2")
                st.success("Done")

    if st.button("Cluade Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_claude_llm()

            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()