import streamlit as st
from create_vector_db import load_environment, load_existing_vector_database
from conversation import setup_conversation_chain, process_query

st.title("Research Publications Chatbot")

# Initialize session state for conversation chain
if "conversation_chain" not in st.session_state:
    # Load environment variables
    load_environment()
    
    # Load the vector database
    vectordb = load_existing_vector_database()
    
    # Setup the conversation chain
    st.session_state.conversation_chain = setup_conversation_chain(vectordb)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask about my research"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(st.session_state.conversation_chain, prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})