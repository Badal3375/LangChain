from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

llm=ChatOllama(
    model="llama3.2"
)
#result = model.invoke("What is the capital of India?")

#print(result.content)