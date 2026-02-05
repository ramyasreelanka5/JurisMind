Legal Assistance Project

This project is a Legal Assistance bot application with a Python backend and a JavaScript-based frontend.

Getting Started

Follow these instructions to get a local copy of the project up and running on your machine.

Prerequisites

Before you begin, ensure you have the following installed:

    Python 3.8+ and pip
    
    Node.js and npm

Install this software for image 

Installation and Setup

1. Install Tesseract OCR

Download and install Tesseract from the official repository: https://github.com/UB-Mannheim/tesseract/wiki .

Important: After installation, you may need to add the Tesseract installation path to your system's environment variables or specify it directly in the backend code. Open backend/app.py and update the following line with your Tesseract path if necessary:

    # Update this path according to your Tesseract installation
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


2. Clone the Repository

First, clone the repository to your local machine.

    git clone https://github.com/kiransrisai/Legal_chatbot.git
    
    cd Legal_chatbot

3. Set Up the Backend

The backend requires Python dependencies and API keys.

Navigate to the backend directory and install the required packages:

    cd backend
    
    pip install -r requirements.txt

Create an environment file:

Create a new file named .env inside the backend folder.

Add your API keys to the .env file:
Open the .env file and add the following lines, replacing the placeholders with your actual keys:
    
    GROQ_API_KEY="your_groq_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"

4. Set Up the Frontend

The frontend requires Node.js modules.

  Navigate to the frontend directory:
  
  # If you are in the backend folder, go back one level first
  
      cd ../frontend

  Install the node modules:
  
    npm install

Running the Application

1. Start the Backend Server

  Navigate to the backend directory:
  
      cd path/to/your/project/backend
  
  Run the application:
  
      python app.py
  
  The backend server will now be running.

2. Start the Frontend Application

  In a new terminal, navigate to the frontend directory:
  
      cd path/to/your/project/frontend
  
  Run the start command (e.g., npm start):
      
      npm start

Your application should now be accessible in your browser.
