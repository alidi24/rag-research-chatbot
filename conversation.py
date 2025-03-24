from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def get_chat_history(chat_history):
    if not chat_history:
        return ""
    
    formatted_chat_history = ""
    for i, message in enumerate(chat_history):
        if i % 2 == 0:  # Human message
            formatted_chat_history += f"Human: {message.content}\n"
        else:  # AI message
            formatted_chat_history += f"AI: {message.content}\n"
    
    return formatted_chat_history

def setup_conversation_chain(vectordb, model_name="gpt-3.5-turbo", temperature=0.7):
    # Initialize LLM
    llm = ChatOpenAI(temperature=temperature, model=model_name)
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Custom prompt template
    template = """You are a friendly and helpful expert assistant.

If the question is a greeting or casual conversation (like "hello", "how are you", "goodbye"), respond naturally and conversationally.

Use the following pieces of context to answer the question at the end.
Pay special attention to publication dates, author names, journal names, and paper titles in the context.
For questions about dates or timelines (like "latest paper" or "papers from 2017"), carefully check the publication years/dates in the metadata of each source.

If asked to summarize or describe papers from a specific year, look for papers with that publication year in their metadata.
If asked about the latest paper, identify the paper with the most recent publication date.

If you don't know the answer or the information isn't in the context, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=get_chat_history,
        combine_docs_chain_kwargs={
            "prompt": prompt
        }
    )
    
    return conversation_chain

def process_query(conversation_chain, query):
    response = conversation_chain({"question": query})
    return response["answer"]