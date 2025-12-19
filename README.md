# ChatBot

This project is a conference assistant chatbot designed to answer user questions about events, bookings, schedules, speakers, and logistics.
The chatbot is built to work out-of-the-box with minimal configuration and is designed to evolve into a fully AI-powered, self-updating system.

🚀 Project Goals

Provide instant answers to conference-related questions

Reduce manual configuration for customers

Automatically update information by reading conference websites

Support both static FAQ data and dynamic AI-based responses

🧠 Current Architecture (Pipeline v1)

The chatbot currently works using a structured data pipeline:

Frontend

chat.html: Simple web-based chat interface

Sends user questions to the backend API

Backend (FastAPI)

main.py: API entry point

chat.py: Handles chat requests

agent.py: Core logic that decides how to answer

prompt.py: Defines system and assistant prompts

retriever.py: Searches structured data (JSON files)

Data Sources

data/conferences/*.json: Conference details (name, date, location, agenda)

data/booking/booking_faq.json: Booking and cancellation FAQs

The chatbot matches user questions against these datasets and returns predefined answers.

🤖 AI Pipeline (Pipeline v2 – In Progress)

The next evolution of the chatbot introduces AI-powered dynamic knowledge:

Crawl one or more conference URLs

Extract readable text from web pages

Build a dynamic knowledge base

Use an LLM (OpenAI) to answer questions based on live website content

Key components:

site_crawler.py: Crawls conference websites and collects URLs

Future loaders: Convert pages into structured documents

LLM integration: Generate answers without manual Q&A setup

This allows customers to only provide a website URL and get a fully functional chatbot.

🛠️ Setup Instructions
1. Clone the repository
git clone <your-repo-url>
cd chatbot

2. Install backend dependencies
pip install -r requirements.txt

3. (Optional) Install frontend dependencies
npm install
npm run dev

4. Run the backend
python main.py

5. Open the chat interface

Open chat.html in your browser or access the running frontend.

📂 Project Structure
chatbot/
│
├── api/
│   └── chat.py
├── assistant/
│   ├── agent.py
│   ├── prompt.py
│   └── retriever.py
├── data/
│   ├── conferences/
│   └── booking/
├── site_crawler.py
├── main.py
├── chat.html
├── requirements.txt
├── pipeline.txt
└── pipelineAI.txt

🔒 Environment Variables

Create a .env file for API keys (future AI integration):

OPENAI_API_KEY=your_api_key_here

🧩 Future Improvements

Vector embeddings for semantic search

Automatic re-crawling and updates

Multi-language support

Voice input/output

User analytics and admin dashboard

📌 Status

✅ Static chatbot working
🚧 AI-powered dynamic pipeline in development
🎯 Goal: Zero-configuration chatbot for conferences
