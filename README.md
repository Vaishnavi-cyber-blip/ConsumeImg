# Flask Backend for Consume Wisely
This Flask application serves as the backend for the Consume Wisely app, enabling product label analysis and claim verification. It leverages AI and external APIs to extract text, analyze claims, and provide real-time news.

## Features
### 1. Text Extraction from Product Labels
- Accepts an image file of a product label.  
- Uses Google Gemini AI to extract and clean text in a formatted manner.  

### 2. Claim Verification
  - Compares user-provided product claims against the extracted text.  
  - Generates an in-depth analysis including:  
          - Claim Accuracy  
          - Ingredient Review  
          - Nutritional Facts Review  
          - Overall Observations  
          - Final Conclusion 

### 3. Real-Time News Search
- Fetches relevant articles and news about the product using the Tavily API.

## 🚀 Endpoints
### 1. /extract_text
- Method: POST  
- Description: Extracts text from a provided image of a product label.  
- Input: An image file (image).  
- Output: Extracted text in JSON format.  

### 2. /claim_analyser
- Method: POST  
- Description: Analyzes the accuracy of a product claim against the extracted label text.  
- Input: JSON with:
     - extractedText: The text extracted from the label.
     - userInput: The claim to be analyzed.
     - productName: (Optional) Name of the product for news search.
- Output: Detailed claim analysis and news related to product.

## Key Components

### APIs and Libraries
- Flask: Web framework for handling requests and routes.
- Google Gemini: AI-powered text extraction and generative content model.
- Tavily API: Fetches real-time news and articles.
- Pillow (PIL): Processes images for text extraction.
- dotenv: Manages environment variables securely.

## Running the Application
### 1. Clone the repository:
``` git clone https://github.com/your-repo/consume-wisely-backend.git
cd consume-wisely-backend
```
### 2. Install dependencies:
``` pip install -r requirements.txt ```

### 3. Set up environment variables in a .env file:
``` GOOGLE_API_KEY=your_google_api_key```  
```TAVILY_API_KEY=your_tavily_api_key ```

### 4. Run the Flask application:
``` python app.py```  
```Access the app at http://localhost:5000.```

## Environment Variables
The application requires the following keys in a .env file:

```GOOGLE_API_KEY: API key for Google Gemini.```  
```TAVILY_API_KEY: API key for Tavily Search.```

## 🌐 Website
Try out the application live:
[Consume Wisely](https://friendly-spork-2.onrender.com/)

## Interface

![consume4](https://github.com/user-attachments/assets/cf6c933c-814c-4305-ae9c-f07e418beb2b)


  
