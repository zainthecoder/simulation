from enum import Enum
from typing import Dict

class PromptType(Enum):
    INITIAL_QUESTIONS = "initial_questions"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    PRICE_COMPARISON = "price_comparison"
    FEATURE_COMPARISON = "feature_comparison"

# Dictionary of prompts for different scenarios
PROMPTS: Dict[PromptType, str] = {
    PromptType.INITIAL_QUESTIONS: """
    User traits:
    - Persona: {persona}
    - Preference: {preference} 
    - Goal: {goal}

    Available products:
    {products}

    Based on the user traits and available products, generate 3-4 general questions to ask about the recommended items.
    Questions should help understand the user's needs better.
    """,

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

    PromptType.PRICE_COMPARISON: """
    User traits:
    - Persona: {persona}
    - Preference: {preference}

    Available products and their prices:
    {product_prices}

    Compare the prices of these phones and recommend the best value option.
    Consider the user's price sensitivity and needs.
    """,

    PromptType.FEATURE_COMPARISON: """
    User traits:
    - Persona: {persona}
    - Preference: {preference}

    Available products and their features:
    {product_features}

    Compare the key features of these phones and recommend the best option.
    Focus on features that matter most to the user's persona.
    """
}

def get_prompt(prompt_type: PromptType, **kwargs) -> str:
    """
    Get a formatted prompt based on the type and provided parameters.
    
    Args:
        prompt_type: Type of prompt to retrieve
        **kwargs: Parameters to format the prompt string
    
    Returns:
        Formatted prompt string
    """
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return PROMPTS[prompt_type].format(**kwargs) 