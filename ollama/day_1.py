from langchain_ollama import OllamaLLM

# Create an Ollama LLM
llm = OllamaLLM(
    model="llama3.2",
    temperature=0.7,
    
)

# Send a prompt
response = llm.invoke("Hello, how are you?")

print(response)
