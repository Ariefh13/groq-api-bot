import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from groq import Groq
import instructor

# Load environment variables
load_dotenv()

def get_groq_api_key():
    """Retrieve the Groq API key from environment variables."""
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment variables.")
    return api_key

class Character(BaseModel):
    name: str
    fact: List[str] = Field(..., description="A list of facts about the subject")

def query_groq(question: str) -> Character:
    """Send a query to the Groq API and return the response."""
    client = Groq(api_key=get_groq_api_key())
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
    
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": question}],
        response_model=Character,
    )
    return response

def main():
    """Main function to interact with the user in a loop."""
    print("Welcome to the Groq API chatbot! Type 'quit' to exit.")
    
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break
        
        try:
            response = query_groq(user_input)
            print(response.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
