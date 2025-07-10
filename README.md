# Rag_Spider_task_1
The code is in 2 files preprocessing.py and retrieval pipeline and webpage.py

preprocessing file loads all the 5 papers and split them into chunks and embedds using **sentence-transformers/all-mpnet-base-v2**
and the embedded content is stored in chromadb vectordatabase which is eventually stored on disk

Retrieval file creates a basic webpage using streamlit and gets input from user and embedds it and similar chunks are retrieved from database along with similarity scores
Similarity scores are also printed in the terminal
Then these chunks are concatenated and passed to the model(**Llama-3.1-8B-Instruct** is being used)
Then the answer from the model is displayed in the webpage
