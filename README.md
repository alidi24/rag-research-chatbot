# Research Publications Chatbot

A RAG-based chatbot that lets you have conversations with your documents - I built it for my research papers. This system uses LangChain, OpenAI and vector search, and shows how to build a conversational AI that actually understands what's in your files.


## Project Structure

- `create_vector_db.py`: Handles document loading and vector database creation
- `retrieval.py`: Sets up the basic question-answering chain (for one-off queries)
- `conversation.py`: Manages conversational memory and chat processing (for multi-turn conversations)
- `cli_app.py`: Command-line interface for the chatbot
- `web_app.py`: Streamlit web interface for the chatbot
- `evaluation.py`: Tools for evaluating the chatbot's performance

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Place your research PDFs in a folder called `docs/` in the project root.

## Usage

### Creating the Vector Database

Before using the chatbot, you need to create a vector database of your research papers:

```python
# Run this standalone script to create the vector database
from create_vector_db import load_environment, load_documents, split_documents, create_vector_database

load_environment()
docs = load_documents()
chunks = split_documents(docs)
create_vector_database(chunks)
```

Or simply run this command in the terminal:
```
python create_vector_db.py
```

### Running the Chatbot

You can run the chatbot through a command-line interface:

```
python cli_app.py
```

Or launch the web interface:

```
streamlit run web_app.py
```

### Evaluation

To evaluate the chatbot's performance, you can modify `evaluation.py` with example questions and expected answers, then run:

```
python evaluation.py
```

## Features

- Semantic search across research publications
- Contextual responses based on document content
- Conversational memory for multi-turn interactions
- Option to deploy as CLI or web application