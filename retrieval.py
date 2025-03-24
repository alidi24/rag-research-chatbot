from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

def format_documents_with_metadata(docs):
    formatted_docs = []
    for doc in docs:
        metadata = doc.metadata
        content = doc.page_content
        
        # Add metadata header to each chunk with clear label for main papers
        header = "## MAIN PAPER METADATA:\n"
        header += f"Title: {metadata.get('title', 'Unknown')}\n"
        header += f"Authors: {metadata.get('authors', 'Unknown')}\n"
        header += f"Year: {metadata.get('year', 'Unknown')}\n"
        header += f"Journal: {metadata.get('journal', 'Unknown')}\n"
        header += f"Source: {metadata.get('source', 'Unknown')}\n\n"
        header += "## CONTENT:\n"
        
        formatted_docs.append(header + content)
    return formatted_docs

def setup_qa_chain(vectordb, model_name="gpt-3.5-turbo", temperature=0.7):
    """
    Create a simple question-answering chain without conversation memory.
    This is useful for one-off questions or when you don't need to maintain
    conversation context. For multi-turn conversations, use the
    setup_conversation_chain function from conversation.py instead.
    """
    llm = ChatOpenAI(temperature=temperature, model=model_name)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    template = """You are a friendly and helpful expert assistant.

If the question is a greeting or casual conversation (like "hello", "how are you", "goodbye"), respond naturally and conversationally.

Use the following pieces of context to answer the question at the end.
Pay special attention to publication dates, author names, journal names, and paper titles in the context.
For questions about dates or timelines (like "latest paper" or "papers from 2017"), carefully check the publication years/dates in the metadata of each source.

If you don't know the answer or the information isn't in the context, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "document_separator": "\n\n"
        }
    )
    
    # Apply document formatting
    original_invoke = qa_chain.invoke
    
    def formatted_invoke(query):
        # Get the response
        response = original_invoke(query)
        
        # Format the source documents if they exist
        if "source_documents" in response:
            response["formatted_sources"] = format_documents_with_metadata(response["source_documents"])
        
        return response
    
    # Replace the invoke method
    qa_chain.invoke = formatted_invoke
    
    return qa_chain

def process_single_query(qa_chain, query):
    """Process a one-off query using the QA chain without retaining conversation history"""
    response = qa_chain({"query": query})
    return response["result"]