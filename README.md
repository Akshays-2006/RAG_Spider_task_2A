# Rag_Spider_task_1
The code is in 2 files preprocessing.py and retrieval pipeline and webpage.py
**Preprocessing.py**
Loads 5 research papers

Splits them into chunks.

Embeds the chunks using: **sentence-transformers/all-mpnet-base-v2**.

Stores the embeddings in ChromaDB (disk-persistent vector database).

**Retrieval_pipeline_and_webpage.py:**

Uses **Streamlit** to build a simple web interface.

Takes question from the user using webpage.

Embeds the question using the same model (all-mpnet-base-v2).

Retrieves top similar chunks from ChromaDB.

Prints similarity scores in the terminal.

Concatenates the retrieved chunks and sends them to **LLaMA 3.1 8B Instruct** for answer generation.

Displays the final answer on the webpage.
