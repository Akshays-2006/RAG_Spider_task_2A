#This file imports necessary libraries and processes the 5 papers and it is then converted into chunks and 
#then embedded using all-mpnet v2 model and stored in chromadb vectordatabase on disk
#which can then be used to retrieve relevant information efficiently

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store=Chroma(
    collection_name="spider_task",
    embedding_function=embedding_function,
    persist_directory="C:\\Users\\aksha\\Desktop\\Spider task 2 papers\\chromadb"
)

documents=["BERT Pre training of Deep Bidirectional Transformers for Language Understanding.pdf","Attention Is All You Need.pdf","Contrastive LanguageImage Pretraining with Knowledge Graphs.pdf","GPT3 Language Models are Few-Shot Learners.pdf","LLaMA Open and Efficient Foundation Language Models.pdf"]
entire_content=""
no_page=0
doc_id=0
page_id=0
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=2000, chunk_overlap=100)
#Documents are loaded one by one and then it is divided into chunks of size 2000 and overlap of 100 
#then embedded and ids are added and stored in chromadb
for file_name in documents:
    path=os.path.join("C:\\Users\\aksha\\Desktop\\Spider task 2 papers",file_name)
    loader = PyPDFLoader(path)
    pages = loader.load()
    combined_text = "\n".join([p.page_content for p in pages if p.page_content])
    texts = text_splitter.split_text(combined_text)
    chunks = [Document(page_content=chunk) for chunk in texts]
    ids = [f"{doc_id}_{page_id}_{chunk_idx}" for chunk_idx in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"{file_name} was divided into {len(chunks)} chunks",end="\n\n")

print(vector_store)
