import os
import time
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymongo import MongoClient
import google.generativeai as genai
from dotenv import load_dotenv
import webbrowser
import threading
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
PDF_DIRECTORY = "supportpdfs"
FAISS_INDEX_PATH = "faiss_index"

# Load .env file if you are using it
load_dotenv()
# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB connection 
client = MongoClient("mongodb://localhost:27017/")
chat_db = client["socialiq_chatbot"]

# Function to ensure collection exists
def ensure_collection():
    collection_names = chat_db.list_collection_names()
    if "user_conversations" not in collection_names:
        # Insert a dummy document to create the collection
        chat_db["user_conversations"].insert_one({"_init": True})
        # Optionally delete the dummy document
        chat_db["user_conversations"].delete_one({"_init": True})

chat_collection = chat_db["user_conversations"]

vectorstore = None

def create_faiss_index():
    """Loads PDFs, splits text, and stores embeddings in FAISS."""
    print("📄 Loading PDFs...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()

    if not docs:
        print("❌ No PDFs found in directory:", PDF_DIRECTORY)
        return

    print(f"✅ Loaded {len(docs)} documents. Splitting text...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    if not documents:
        print("Text splitter returned no chunks.")
        return

    print(f"✂️ Split into {len(documents)} chunks. Generating embeddings...")

    try:
        # Using Google embeddings instead of OpenAI
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        print("🔥 Error during embedding or FAISS index creation:", str(e))
        return

    vectorstore.save_local(FAISS_INDEX_PATH)
    print("✅ FAISS index created and saved locally.")

# Initialize FAISS index
def initialize_vectorstore():
    global vectorstore
    if not os.path.exists(FAISS_INDEX_PATH):
        create_faiss_index()
    else:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ FAISS index loaded successfully.")

def retrieve_relevant_context(query):
    """Fetches relevant document chunks from FAISS for a given query."""
    if vectorstore is None:
        return "No context available."
    
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

def generate_response_with_google(prompt):
    """Generates AI responses using Google Generative AI."""
    try:
        # Updated model names - using current stable models
        model = genai.GenerativeModel('gemini-1.5-flash')  # Changed from 'gemini-pro'
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"🔥 Error generating response: {e}")
        # Fallback to different model if needed
        try:
            print("Trying fallback model: gemini-2.0-flash")
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            print(f"🔥 Error with fallback model: {e2}")
            return "Sorry, I encountered an error while processing your request."

def generate_quick_suggestions(response_text):
    """Generate quick suggestions based on the response."""
    try:
        suggestion_prompt = f"""
        Based on this response: "{response_text[:500]}..."
        Generate 3 short follow-up questions (max 10 words each) that users might ask.
        Return as a simple list, one per line.
        """
        
        # Updated model names - using current stable models
        model = genai.GenerativeModel('gemini-1.5-flash')  # Changed from 'gemini-pro'
        suggestions_response = model.generate_content(suggestion_prompt)
        suggestions = suggestions_response.text.strip().split('\n')
        
        # Clean and limit suggestions
        clean_suggestions = []
        for suggestion in suggestions[:3]:
            clean_suggestion = suggestion.strip().lstrip('1234567890.-• ')
            if clean_suggestion and len(clean_suggestion) <= 50:
                clean_suggestions.append(clean_suggestion)
        
        return clean_suggestions[:3]
    except Exception as e:
        print(f"🔥 Error generating suggestions: {e}")
        # Fallback to different model if needed
        try:
            print("Trying fallback model for suggestions: gemini-2.0-flash")
            model = genai.GenerativeModel('gemini-2.0-flash')
            suggestions_response = model.generate_content(suggestion_prompt)
            suggestions = suggestions_response.text.strip().split('\n')
            
            # Clean and limit suggestions
            clean_suggestions = []
            for suggestion in suggestions[:3]:
                clean_suggestion = suggestion.strip().lstrip('1234567890.-• ')
                if clean_suggestion and len(clean_suggestion) <= 50:
                    clean_suggestions.append(clean_suggestion)
            
            return clean_suggestions[:3]
        except Exception as e2:
            print(f"🔥 Error with fallback model for suggestions: {e2}")
            return ["How can I help you more?", "Any other questions?", "What else would you like to know?"]

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat requests via REST API."""
    try:
        data = request.get_json()
        
        email = data.get("email")
        question = data.get("question", "")

        if not email or not question:
            return jsonify({"error": "Email and question are required!"}), 400

        print(f"Received question from {email}: {question}")

        # Get conversation history
        conversation_entry = chat_collection.find_one({"email": email})
        
        if conversation_entry and "conversation" in conversation_entry:
            last_5_conversations = conversation_entry["conversation"][-5:]
        else:
            last_5_conversations = []

        # Format conversation history
        conversation_history = "\n".join(
            f"User: {entry['question']}\nBot: {entry['answer']}"
            for entry in last_5_conversations
        )

        # Retrieve relevant context from FAISS
        relevant_context = retrieve_relevant_context(question)

        # Construct prompt for Google AI
        prompt = f"""
        You are an AI assistant trained for the "Social IQ" app. Respond to the user helping users with System Working Using provided Context Documents.

        *Ensure Whenever question asked regarding Technology Used in Application of SocialIQ Always Return *Google Gemini API*, *Python as backend*, *MongoDB as Database*,*Flutter as Frontend* Don't check curriculum context for this as that thing mentioned is wrong correct is python and flutter which i given here*

        Context from documents:
        {relevant_context}

        Conversation history:
        {conversation_history}

        User: {question}
        Bot:
        """

        # Generate response using Google AI
        response_text = generate_response_with_google(prompt)

        # Generate quick suggestions
        quick_suggestions = generate_quick_suggestions(response_text)

        # Get timestamp
        timestamp = time.strftime("%d-%m-%Y %I:%M %p")

        # Store in MongoDB
        chat_collection.update_one(
            {"email": email},
            {
                "$push": {
                    "conversation": {
                        "question": question,
                        "answer": response_text,
                        "quickSuggestion": quick_suggestions,
                        "timestamp": timestamp,
                    }
                }
            },
            upsert=True,
        )

        return jsonify({
            "response": response_text,
            "quick_suggestions": quick_suggestions,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"🔥 Error handling chat: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/chat_history/<email>', methods=['GET'])
def get_chat_history(email):
    """Get chat history for a user."""
    try:
        conversation_entry = chat_collection.find_one({"email": email})
        
        if conversation_entry and "conversation" in conversation_entry:
            return jsonify({
                "email": email,
                "conversations": conversation_entry["conversation"]
            })
        else:
            return jsonify({
                "email": email,
                "conversations": []
            })
    
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return jsonify({"error": "Internal server error"}), 500

from flask import send_from_directory
import os

@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Chatbot is running!"})

if __name__ == '__main__':
    # Initialize vectorstore before starting the app
    initialize_vectorstore()

    print("Starting chatbot server...")
    app.run(debug=True, host='0.0.0.0', port=5000)