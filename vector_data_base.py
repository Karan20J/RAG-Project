from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

pdf_loader = PyPDFLoader("Machine learning book .pdf")
docs = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
save_splitter = splitter.split_documents(docs)
embeddings = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=save_splitter,
    embedding=embeddings,
    persist_directory="project_2_store",
)



