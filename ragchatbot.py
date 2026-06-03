import os
import pdfplumber
import streamlit as st

# from env import GEMINI_API_KEY
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    from env import GEMINI_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

st.title("My RAG Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type=["pdf"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )

    import traceback

    try:
        vector_store = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"**Error type:** {type(e).__name__}")
        st.error(f"**Message:** {str(e)}")
        st.error(f"**Cause:** {str(e.__cause__)}")
        st.code(traceback.format_exc())
        st.stop()

    user_question = st.text_input("Type your question here")

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=1024,
        google_api_key=GEMINI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful assistant that answers questions about a PDF document.

Guidelines:
- Provide complete, well explained answers using the context below
- Include relevant details, numbers and explanations to give a thorough response
- If the context mentions related information, include it to give a fuller picture
- Only use information from the provided context - do not use outside knowledge
- Summarize long information, ideally in bullets where needed
- If the information is not in the context, say so politely

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if user_question:
        try:
            response = chain.invoke(user_question)
            st.write(response)
        except Exception as error:
            st.error(f"Failed to generate a response: {error}")