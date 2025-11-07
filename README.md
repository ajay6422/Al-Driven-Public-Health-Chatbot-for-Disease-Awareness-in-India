# Overview Of The Project

The AI-Driven Health Chatbot is an intelligent virtual assistant designed to help users with basic health-related queries.
It leverages artificial intelligence and natural language processing to provide accurate, reliable, and easily understandable medical information.
By interacting with the chatbot, users can receive personalized guidance, symptom-based insights, and preventive care suggestions, helping them make informed decisions about their health.
The goal of this project is to promote accessible healthcare awareness and support users in maintaining a healthy lifestyle through timely and relevant advice.

## Project Layout

### Step 1

We have setup memory to store data by which chatbot will answer your query
we have  load a book(The tale of Encyclopidea) or Pdf of the data 
Afetr that we have made chunks and after that we have made embeddings
We have used a vector database named FAISS to store embeddings

### Step 2

Connect memory with LLM
we have setup llm ( we have use hugging faec for connecting the llm)
After that we have connectd LLM with FAISS

### Step 3

At last we have created a Ui for chatbot by using streamlit
We have load vector data in cache 

## Technologies Used

1 Langchain (It is an AI framework for llm applications)

2 Hugging Face (It is a hub for AI and ML where people and companies push models and anyone can download free models in theri projects)

3 Mistral (It is model of LLM used from hugging face)

4 FAISS (It is a vector database used to store data it is cloud based so not uses storage of our pc)

5 Steamlit (It is used for frontend user interface and turn your Python scripts into web apps without using HTML CSS and react)

6 Python coading language
