import io
import boto3
import streamlit as st
from pypdf import PdfReader

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate


## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)



def data_ingestion(pdf):
    pdf_reader = PdfReader(io.BytesIO(pdf.getvalue()))
    
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    
    texts = text_splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]
    
    return docs


def get_vector_store(docs, index_name):
    if not docs:
        raise ValueError("The document list is empty. Please check the data ingestion process.")
    
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local(index_name)

def get_llm():
    ## create the Anthropic Model
    llm=Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0", 
        client=bedrock,
        model_kwargs={'max_gen_len':512}
    )
    return llm


prompt_template = """
Human: You are provided with context from one or two documents. 
Use the information from one or both documents to answer the question at the end. 
More specifically, if you only have one, then only answer given that document.
But if you have two, then answer as if you are comparing or contrasting them. 
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



def get_response_llm(llm, vectorstore_faiss1, vectorstore_faiss2, query, use_both=False, use_1_or_2=1):
    """Get the response from the LLM
    Load the vector stores and get the documents. 
    Then input the contexts into the prompt depending on the use case
    Finally input the prompt into the LLM
    """
    if use_both:
        retriever1 = vectorstore_faiss1.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        retriever2 = vectorstore_faiss2.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        docs1 = retriever1.get_relevant_documents(query)
        docs2 = retriever2.get_relevant_documents(query)

        context1 = "\n".join([doc.page_content for doc in docs1])
        context2 = "\n".join([doc.page_content for doc in docs2])

    else:
        if use_1_or_2==1:
            retriever1 = vectorstore_faiss1.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            docs1 = retriever1.get_relevant_documents(query)
            context1 = "\n".join([doc.page_content for doc in docs1])
            context2 = ""
        elif use_1_or_2==2:
            retriever2 = vectorstore_faiss2.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            docs2 = retriever2.get_relevant_documents(query)
            context1 = ""
            context2 = "\n".join([doc.page_content for doc in docs2])

    prompt = PROMPT.format(context1=context1, context2=context2, question=query)
    answer = llm(prompt)
    return answer

def chat_with_documents(user_question, use_both, use_1_or_2=None):
    """
    If the user clicks on the button to chat with both documents, then we'll load both.
    Otherwise we load the one they want to chat with only.
    """
    try:
        with st.spinner("Processing..."):
            llm = get_llm()
            if use_both:
                vectorstore_faiss1 = FAISS.load_local("faiss_index1", bedrock_embeddings, allow_dangerous_deserialization=True)
                vectorstore_faiss2 = FAISS.load_local("faiss_index2", bedrock_embeddings, allow_dangerous_deserialization=True)
                response = get_response_llm(llm, vectorstore_faiss1, vectorstore_faiss2, user_question, use_both=True)
            elif use_1_or_2 == 1:
                vectorstore_faiss1 = FAISS.load_local("faiss_index1", bedrock_embeddings, allow_dangerous_deserialization=True)
                response = get_response_llm(llm, vectorstore_faiss1, None, user_question, use_both=False, use_1_or_2=1)
            elif use_1_or_2 == 2:
                vectorstore_faiss2 = FAISS.load_local("faiss_index2", bedrock_embeddings, allow_dangerous_deserialization=True)
                response = get_response_llm(llm, None, vectorstore_faiss2, user_question, use_both=False, use_1_or_2=2)
        return response
    
    except UnboundLocalError:
        st.write(f"Please upload {'both documents' if use_both else f'document {use_1_or_2}'}")
    
    except Exception as e:
        error_type = type(e).__name__
        st.write(f"An error occurred: {e} (Error type: {error_type})")
    
    return ""


def main():
    st.set_page_config(page_title="Chatting with two PDFs", page_icon=":books:")
    st.header("Chat and compare two PDFs")

    user_question = st.text_input("Ask a question from the PDF files")
    response = ""

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

    col1, col2, col3 = st.columns(3)

    if col1.button("Chat with both Documents"):
        response = chat_with_documents(user_question, use_both=True)

    if col2.button("Chat with Document 1"):
        response = chat_with_documents(user_question, use_both=False, use_1_or_2=1)

    if col3.button("Chat with Document 2"):
        response = chat_with_documents(user_question, use_both=False, use_1_or_2=2)

    st.write(response)



if __name__ == "__main__":
    main()