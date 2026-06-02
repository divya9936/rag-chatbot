import pdfplumber
import streamlit as st
from langchain_classic import text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title(' My Rag Chatbot')
with st.sidebar:
    st.title('Your Documents')
    file = st.file_uploader('Upload a PDF file and start asking questions', type=['pdf'])

# Extract content from the pdf and chunk it
if file is not None:
    # extract text from it
    with pdfplumber.open(file) as pdf:
        text = " "
        for page in pdf.pages:
            text = text + page.extract_text()+ "\n"
    #st.write(text)

    #split text into chunks
    text_splitter =  RecursiveCharacterTextSplitter(
        separators= ["\n\n", "\n" , "," , " " , ""],
        chunk_size= 1000,
        chunk_overlap= 200
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)