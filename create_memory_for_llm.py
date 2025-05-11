from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_PIPhuggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


Data_path="Data/"

def load_data_files(data):
    Documents=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = Documents.load()
    return documents

documents=load_data_files(Data_path)


def create_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(doc)
    return chunks

text_chunks=create_chunks(documents)


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
embeddings = create_embeddings()
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
