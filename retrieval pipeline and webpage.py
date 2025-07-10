#This file is creats a webpage using streamlit and then getting input from user 
#and then its embedded and top 3 relevant chunks are retrieved from chroma vectordatabase with similarity score
#these chunks are concatenated and passed to llm model which is Llama-3.1-8B-Instruct
#and the response is displayed in the webpage also similarity score is logged in terminal\

import streamlit as st
import base64
#Basic webpage is created using streamlit
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    css = f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
        background-size: 20% 40%;
        opacity: 0.4;  
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("C:\\Users\\aksha\\Downloads\\images-removebg-preview.png")

st.title("_RAG PIPELINE_")
st.write('''This is a RAG pipeline which digests data from the given 5 research papers and then retrives
         relevant content from the papers and then it is passed to llama model to answer the question''')


import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})
try:
    vector_store = Chroma(
        collection_name="spider_task",
        embedding_function=embedding_function,
        persist_directory="C:\\Users\\aksha\\Desktop\\Spider task 2 papers\\chromadb"
    )
except Exception as e:
    st.error(f"Error initializing vector store: {e}")
    st.stop()
#Input box is created which gets the question from user
s=st.chat_input(placeholder="Ask a question about the research papers...")
#input question is embedded and similar chunks are retrieved from the vectordatabase
if s:
    with st.spinner("Retrieving and answering your question..."):
        similar_content=vector_store.similarity_search_with_score(
        s,
        k=3
    )

        context=""
        print("similarity of score of retrieved contents: ")
        for content,score in similar_content:
            context+=content.page_content
        print(score)
        print(len(content.page_content))
        print(len(context))


        client = InferenceClient(
    provider="nebius",
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)
 
        question=s
#A prompt is created consisting of Question along with retrieved chunks 
        prompt=f"""Use only the following retrieved context to answer the question. 
You may infer possible approaches based on the context, even if they are not explicitly mentioned.
If there is truly no relevant information or clues in the context, then say "I don't know."

Question:{question}

Context:{context}

"""

        completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
        )
    st.success("Done!")
    st.markdown(completion.choices[0].message.content)
    #Finally the answer from chatbot is displayed
