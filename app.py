import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import easyocr
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import re

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Retrieve API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize Groq LLaMA model
llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

# Tavily search implementation
# Tavily search implementation
def tavily_search(search_query: str):
    """Search the internet about a given topic using the Tavily API and return relevant results."""
    try:
        # Initialize the Tavily API wrapper
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        tavily_tool = TavilySearchResults(api_wrapper=api_wrapper)
        
        # Perform search using the Tavily tool
        results = tavily_tool.run(search_query)
        
        # Check if results were returned, log and handle empty results
        if not results or len(results) == 0:
            logging.warning(f"No results found for query: {search_query}")
            return {"error": "No results found"}

        # Store unique results to prevent duplication
        seen_urls = set()
        formatted_results = []

        # Iterate over results, filtering out duplicates and formatting the output
        for result in results:
            url = result.get('url', '')
            content = result.get('content', '')

            if url and url not in seen_urls:
                seen_urls.add(url)
                formatted_results.append({
                    "content": content[:200] + "..." if len(content) > 200 else content,  # Limit content size
                    "url": url
                })

        # Return formatted results
        return formatted_results

    except Exception as e:
        logging.error(f"Error with Tavily search: {e}")
        return {"error": f"Failed to search Tavily API: {str(e)}"}

# Endpoint for extracting text from an image
@app.route('/extract_text', methods=['POST'])
def extract_text():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file
        image = Image.open(io.BytesIO(image_file.read()))
        image = image.convert('RGB')
        image_np = np.array(image)

        # Use EasyOCR to do OCR on the image
        extracted_text_list = reader.readtext(image_np, detail=0)

        # Combine the extracted text into a single string
        extracted_text = " ".join(extracted_text_list)

        # Prepare the prompt for Groq LLaMA processing
        prompt = f"Format and clean the following extracted text: {extracted_text}"

        # Use Groq's API to process the extracted text with the LLaMA model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            
        )

        # Get the processed text from the response
        processed_text = chat_completion.choices[0].message.content

        return jsonify({'text': processed_text})
    
    except Exception as e:
        app.logger.error(f"Error in extract_text: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint for analyzing claims against extracted text
@app.route('/claim_analyser', methods=['POST'])
def claim_analyser():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        extracted_text = data.get('extractedText', '')
        user_input = data.get('userInput', '')
        product_name = data.get('productName', '')

        if not extracted_text or not user_input:
            return jsonify({'error': 'Both extracted text and user input are required'}), 400

        # Few-shot prompting: Provide examples of how to structure the output
        prompt = f"""
        Compare the following product claims against the extracted text: "{extracted_text}". The claim to verify is: "{user_input}". Is the claim accurate? Provide a detailed and interactive analysis using the format below:

        **Example Analysis:**
        ---
        **Claim:**
        This product is 100% organic.

        **🔍 Claim Accuracy:**
        - The product contains some ingredients that are not certified organic.
        - The packaging does not specify certification for organic ingredients.
        - **🟡 Verdict**: This claim is **inaccurate** as not all ingredients are organic.

        **🧪 Ingredient Review:**
        - No evidence of added sugar in the ingredients list.
        - Nutritional facts confirm no sugars listed.
        - **🟢 Verdict**: The product is free from added sugars.

        **📊 Nutritional Facts Review:**
        - The product is low in carbohydrates and fats, supporting the claim of being healthy.
        - **🟡 Verdict**: The product is healthy but not organic as claimed.

        **🔍 Overall Observation:**
        - The product appears to be a processed fruit drink with a combination of fruit concentrates and purees, rather than being made with 100% original fruits.
        - The added sugar and processing steps do not align with the claim of being 100% pure or natural.

        **⚖️ Conclusion:**
        - The claim "Made with 100 percent original fruits" is **inaccurate**.
        - The product's ingredients and nutritional facts do not support this claim.
        - The presence of processed fruit concentrates, purees, and added sugar contradicts the idea of it being made with 100% original fruits.

        Please follow this interactive format for your analysis, using icons, section headers, and clear conclusions.

        ### Now analyze the claim: "{user_input}"
        ---
        """
        # Use Groq's API (or LLaMA API) to process the claim analysis with the LLaMA model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            max_tokens=500
        )

        # Get the analysis result from the response
        analysis_text = chat_completion.choices[0].message.content
        print("Full analysis_text from LLM:", analysis_text)

        # Now parse the response text to match the sections
        # Using simpler regex patterns to capture content after symbols
        claim_match = re.search(r'\*\*Claim:\*\*\n(.*?)\n', analysis_text, re.S)
        claim_accuracy_match = re.search(r'\*\*🔍 Claim Accuracy:\*\*\n(.*?)\n\*\*🧪', analysis_text, re.S)
        ingredient_review_match = re.search(r'\*\*🧪 Ingredient Review:\*\*\n(.*?)\n\*\*📊', analysis_text, re.S)
        nutritional_facts_match = re.search(r'\*\*📊 Nutritional Facts Review:\*\*\n(.*?)\n\*\*🔍', analysis_text, re.S)
        overall_observation_match = re.search(r'\*\*🔍 Overall Observation:\*\*\n(.*?)\n\*\*⚖️', analysis_text, re.S)
        conclusion_match = re.search(r'\*\*⚖️ Conclusion:\*\*\n(.*)', analysis_text, re.S)

        # Extract the groups and assign them to your response
        claim = claim_match.group(1).strip() if claim_match else ''
        claim_accuracy = claim_accuracy_match.group(1).strip() if claim_accuracy_match else ''
        ingredient_review = ingredient_review_match.group(1).strip() if ingredient_review_match else ''
        nutritional_facts = nutritional_facts_match.group(1).strip() if nutritional_facts_match else ''
        overall_observation = overall_observation_match.group(1).strip() if overall_observation_match else ''
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ''

        # Create the response dictionary
        analysis_response = {
            "claim": claim,
            "claimAccuracy": claim_accuracy,
            "ingredientReview": ingredient_review,
            "nutritionalFactsReview": nutritional_facts,
            "overallObservation": overall_observation,
            "conclusion": conclusion
        }
        if product_name:
            news_results = tavily_search(product_name)
        else:
            news_results = "No product name provided, cannot search for news."

        # Return the JSON response
        return jsonify({'analysis': analysis_response, 'newsStatus': news_results})
    
        

        # Return the result and any related news in JSON format
        
    
    except Exception as e:
        logging.error(f"Error in claim_analyser: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred while analyzing the claim: {str(e)}"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


