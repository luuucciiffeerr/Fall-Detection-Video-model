from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle

def build_vectorstore(documents, save_path="vectorstore/faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in documents:
        chunks += splitter.split_text(doc)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_vectorstore(save_path="vectorstore/faiss_index"):
    import pickle
    with open(save_path, "rb") as f:
        return pickle.load(f)
