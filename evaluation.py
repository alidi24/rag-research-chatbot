from langchain.evaluation.qa import QAEvalChain
from langchain.chat_models import ChatOpenAI
from create_vector_db import load_environment, load_existing_vector_database
from retrieval import setup_qa_chain

def evaluate_qa_chain(qa_chain, examples=None):
    if examples is None:
        examples = [
            {"query": "What are the key findings in the paper?", 
             "answer": "The expected ground truth answer..."}
        ]
    
    # Run predictions
    predictions = qa_chain.apply(examples)
    
    # Set up evaluation
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)
    
    # Print results
    for i, grade in enumerate(graded_outputs):
        print(f"Example {i} keys: {grade.keys()}")
        
        # Try to access the evaluation result
        if "text" in grade:
            print(f"Example {i}: {grade['text']}")
        elif "feedback" in grade:
            print(f"Example {i}: {grade['feedback']}")
        elif "score" in grade:
            print(f"Example {i}: {grade['score']}")
        else:
            print(f"Example {i} evaluation: {grade}")

    return graded_outputs

if __name__ == "__main__":
    # Load environment
    load_environment()
    
    # Load vector database
    vectordb = load_existing_vector_database()
    
    # Setup QA chain
    qa_chain = setup_qa_chain(vectordb)
    
    # Define evaluation examples
    examples = [
        {
            "query": "What is the anoamly detection method used in the paper?",
            "answer": "The answer should discuss autoencoder model used for anomaly detection in the paper"
        }
    ]
    
    # Run evaluation
    print("Running evaluation on question-answering chain...")
    evaluate_qa_chain(qa_chain, examples)