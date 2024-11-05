from langchain_community.llms import Ollama
import pandas as pd

# Load the LLM model 
llm = Ollama(model="llama2")

# Function to generate questions and answers
def generate_qa_dataset(num_questions):
    questions = []
    answers = []

    for i in range(num_questions):
        # Generate a question
        print("Question",i)
        question_prompt = "Generate a short question about the Ramayana in 10 words, focusing on its characters, events, or themes."
        question = llm(question_prompt).strip()
        
        if question:  # Ensure a question is generated
            questions.append(question)
            
            # Generate an answer for the question
            answer_prompt = f"Answer the following question about the Ramayana in crisp in 30 words: {question}"
            answer = llm(answer_prompt).strip()
            answers.append(answer)
        else:
            # Handle cases where the question generation fails
            questions.append("Failed to generate question")
            answers.append("No answer available")
    
    return questions, answers

# Main execution
if __name__ == "__main__":
    num_questions = 100

    # Generate the Q&A dataset
    questions, answers = generate_qa_dataset(num_questions)

    # Create a DataFrame and save to CSV
    data = pd.DataFrame({'Question': questions, 'Answer': answers})
    data.to_csv('ramayana_qa_dataset.csv', index=False)
    
    print("Q&A dataset created and saved as ramayana_qa_dataset.csv")
