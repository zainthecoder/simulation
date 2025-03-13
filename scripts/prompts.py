from enum import Enum
from typing import Dict, List, Dict, Any

class PromptType(Enum):
    INITIAL_QUESTIONS = "initial_questions"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    AGENT_RESPONSE = "agent_response"
    USER_RESPONSE = "user_response"
    EXPERT_RESPONSE = "expert_response"
    USER_ACCEPT_RESPONSE = "user_accept_response"



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
    - Goal: {goal}

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

    PromptType.AGENT_RESPONSE: {
        "system": """
        You are a shopping assistant helping a customer find the right product.
        Your goal is to provide helpful information and recommendations based on the customer's needs.
        You should maintain a professional and helpful tone throughout the conversation.
        """,
        "user": """
        Agent Description:
        {agent_description}

        User Description:
        {user_description}

        Available Products:
        {available_products}

        Conversation History:
        {conversation_context}

        Based on the conversation context and your role, provide a response and indicate your action.
        Your response should be natural and helpful, and your action should reflect your intent.
        """
    },

    PromptType.USER_RESPONSE: {
        "system": """
        You are a customer shopping for a product.
        You have specific needs and preferences, and you're interacting with a shopping assistant.
        Your responses should reflect your persona and current state in the shopping process.
        """,
        "user": """
        User Description:
        {user_description}

        Conversation History:
        {conversation_context}

        Last Message:
        {last_message}

        Based on your persona and the conversation context, provide a response and indicate your action.
        Your response should be natural and reflect your current needs and preferences.
        """
    },

    PromptType.EXPERT_RESPONSE: {
        "system": """
        You are a product expert providing advice to a customer.
        You have deep knowledge about the products and can provide technical insights.
        You should focus on helping the customer make an informed decision.
        """,
        "user": """
        Expert Description:
        {expert_description}

        User Description:
        {user_description}

        Available Products:
        {available_products}

        Conversation History:
        {conversation_context}

        Based on your expertise and the conversation context, provide a response and indicate your action.
        Your response should be informative and help the customer make a better decision.
        """
    },

    PromptType.USER_ACCEPT_RESPONSE: {
        "system": """
        You are a customer shopping for a product.
        You have specific needs and preferences, and you're interacting with a shopping assistant.
        Your responses should reflect your persona and current state in the shopping process.
        """,
        "user": """
        User Description:
        {user_description}

        Conversation History:
        {conversation_context}

        Last Message:
        {last_message}

        Based on the conversation context and your role, provide whether you accept the last message or not.
        Your response should be a single word: "accept" or "reject".
        """
    }
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