from enum import Enum
from typing import Dict, List, Dict, Any

class PromptType(Enum):
    INITIAL_QUESTIONS = "initial_questions"
    PRODUCT_RECOMMENDATION = "product_recommendation"



# Dictionary of prompts for different scenarios
PROMPTS: Dict[PromptType, str] = {
    PromptType.INITIAL_QUESTIONS: {
    "system":"""
    You are customer and you are shopping for a product.
    You have certain traits and preferences.
    You are given a list of products and their features.
    You need to select the most relevant product and generate a general question to ask about the product.
    Questions should help understand the user's needs better.
    """,
    "user": """
    User traits:
    - Persona: {persona}
    - Preference: {preference} 
    

    Available products:
    {products}
    """},

    PromptType.PRODUCT_RECOMMENDATION: """
    User traits:
    - Persona: {persona}
    - Preference: {preference}
    - Goal: {goal}

    Available products and their reviews:
    {product_reviews}

    Based on the user's profile and product reviews, recommend the most suitable phone.
    Provide a brief explanation for your recommendation.
    """,
}

def get_prompt(prompt_type: PromptType, **kwargs) -> List[Dict[str, str]]:
    """
    Get formatted messages including system message and user prompt.
    
    Args:
        prompt_type: Type of prompt to retrieve
        **kwargs: Parameters to format the prompt string
    
    Returns:
        List of message dictionaries with role and content
    """
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return [
        {"role": "system", "content": PROMPTS[prompt_type]["system"]},  
        {"role": "user", "content": PROMPTS[prompt_type]["user"].format(**kwargs)}
    ] 