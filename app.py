
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
import re
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-pro')


# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://friendly-spork-2.onrender.com"}})


# Real-time news/articles search related to product
def tavily_search(search_query: str):
    """Search the internet about a given topic using the Tavily API and return relevant results."""
    try:
        # Initialize the Tavily API wrapper
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        tavily_tool = TavilySearchResults(api_wrapper=api_wrapper)
        
        # Perform search using the Tavily tool
        results = tavily_tool.run(search_query)
        
        if not results or len(results) == 0:
            logging.warning(f"No results found for query: {search_query}")
            return {"error": "No results found"}

        seen_urls = set()
        formatted_results = []

        for result in results:
            url = result.get('url', '')
            content = result.get('content', '')

            if url and url not in seen_urls:
                seen_urls.add(url)
                formatted_results.append({
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "url": url
                })

        return formatted_results

    except Exception as e:
        logging.error(f"Error with Tavily search: {e}")
        return {"error": f"Failed to search Tavily API: {str(e)}"}

# Endpoint for extracting text from an image
@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Get the image file from the request
    image_file = request.files['image']

    # Open the image directly from the uploaded file (in memory)
    img = Image.open(io.BytesIO(image_file.read()))

    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Generate content (extract text from image)
    try:
        response = model.generate_content(["Extract and clean the text in formatted manner.", img], stream=True)
        response.resolve()
        extracted_text = response.text

        img.close()
        return jsonify({'text': extracted_text}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
            

# Endpoint for analyzing claims against extracted text
@app.route('/claim_analyser', methods=['POST'])
def claim_analyser():
    
    try:
        data = request.get_json()
        extracted_text = data.get('extractedText', '')
        user_input = data.get('userInput', '')
        product_name = data.get('productName', '')

        if not extracted_text or not user_input:
            return jsonify({'error': 'Both extracted text and user input are required'}), 400

        prompt = f"""
        You are given the following product claim: "{user_input}".
        The product description extracted from its label is: "{extracted_text}".

        ### Instructions:
        1. Analyze the accuracy of the claim based on the extracted text.
        2. Provide an in-depth, step-by-step breakdown with facts.
        3. Ensure the analysis includes:
            - Claim accuracy check
            - Ingredient review
            - Nutritional facts review
            - Overall observations
            - Final conclusion with a verdict (Accurate, Partially Accurate, or Inaccurate)
        4. Use the format given to display the output.


        **Example Analysis:**
        ---
        **Claim:**
        "{user_input}"

        **🔍 Claim Accuracy:**
        - Bullet-pointed facts based on comparison with extracted text.

        **🧪 Ingredient Review:**
        - Bullet-pointed review of the ingredients.

        **📊 Nutritional Facts Review:**
        - Bullet-pointed nutritional facts related to the claim.

        **🔍 Overall Observation:**
        - General summary of findings.

        **⚖️ Conclusion:**
        - Verdict and reasons for the accuracy of the claim.

        ### Now analyze the claim: "{user_input}"
        ---
        """
       
        response = model.generate_content(prompt)
        analysis_text = response.text
        print(analysis_text)

        # analysis_text = chat_completion.choices[0].message.content

        claim_match = re.search(r'\*\*Claim:\*\*\n(.*?)\n', analysis_text, re.S)
        claim_accuracy_match = re.search(r'\*\*🔍 Claim Accuracy:\*\*\n(.*?)\n\*\*🧪', analysis_text, re.S)
        ingredient_review_match = re.search(r'\*\*🧪 Ingredient Review:\*\*\n(.*?)\n\*\*📊', analysis_text, re.S)
        nutritional_facts_match = re.search(r'\*\*📊 Nutritional Facts Review:\*\*\n(.*?)\n\*\*🔍', analysis_text, re.S)
        overall_observation_match = re.search(r'\*\*🔍 Overall Observation:\*\*\n(.*?)\n\*\*⚖️', analysis_text, re.S)
        conclusion_match = re.search(r'\*\*⚖️ Conclusion:\*\*\n(.*)', analysis_text, re.S)

        claim = claim_match.group(1).strip() if claim_match else ''
        claim_accuracy = claim_accuracy_match.group(1).strip() if claim_accuracy_match else ''
        ingredient_review = ingredient_review_match.group(1).strip() if ingredient_review_match else ''
        nutritional_facts = nutritional_facts_match.group(1).strip() if nutritional_facts_match else ''
        overall_observation = overall_observation_match.group(1).strip() if overall_observation_match else ''
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ''

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

        return jsonify({'analysis': analysis_response, 'newsStatus': news_results})
    
    except Exception as e:
        logging.error(f"Error in claim_analyser: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred while analyzing the claim: {str(e)}"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
