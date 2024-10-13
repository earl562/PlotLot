#Deal or no deal tool
#Variance propasal tool

import os
import re
import logging
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
load_dotenv()

# Environment Variables
LLAMAPARSE_API_KEY = os.getenv("LLAMA_API_KEY")

# Initialize OpenAI Embedding Model with API Key

def extract_number(text):
    """
    Extract the first integer found in a text string.
    
    Args:
        text (str): Input text.
    
    Returns:
        int or None: Extracted number or None if not found.
    """
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

def calculate_max_allowable_units(property_details):
    """
    Calculate the maximum allowable units for a given property based on zoning ordinance docs.
    """
    return 
    
def streamline_variance_application(property_details, economic_factors):
    """
    Streamline the variance application process based on property details and economic indicators.

    Args:
        property_details (dict): Dictionary containing property attributes.
        economic_factors (dict): Dictionary containing economic indicators.

    Returns:
        str: Variance application recommendation or error message.
    """
    try:
        # Initialize LlamaParse with o1-mini model
        parser = LlamaParse(api_key=LLAMAPARSE_API_KEY, model="o1-mini")
        
        # Prepare the prompt
        prompt = f"""
        Given the following property address, construct a variance rezoning application that will help me rezone properties of interest:
        Property Address: {property_details['address']}
        """
        
        # Parse the response
        response = parser.parse(prompt)
        
        if not response:
            raise ValueError("No response received from LlamaParse.")
        
        logging.info(f"Variance recommendation for {property_details['address']}: {response}")
        return response
    
    except Exception as e:
        logging.error(f"Error in streamline_variance_application: {e}")
        return "An error occurred while processing the variance application."
