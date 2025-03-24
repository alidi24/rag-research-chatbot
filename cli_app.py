from create_vector_db import load_environment, load_existing_vector_database
from conversation import setup_conversation_chain, process_query

def chat_interface(conversation_chain):
    print("Research Publications Chatbot (type 'exit' to end)")
    print("--------------------------------------------------")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        
        answer = process_query(conversation_chain, query)
        print(f"\nChatbot: {answer}")

if __name__ == "__main__":
    # Load environment variables
    load_environment()
    
    # Load the vector database
    vectordb = load_existing_vector_database()
    
    # Setup the conversation chain
    conversation_chain = setup_conversation_chain(vectordb)
    
    # Run the interface
    chat_interface(conversation_chain)