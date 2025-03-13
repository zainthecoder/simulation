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
        "system": """
    You are a customer and you are shopping for a product.
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
    """,
    },
    PromptType.AGENT_RESPONSE: {
        "system": """
        You are a sales assistant helping a customer find the right product.
        Your goal is to provide helpful information and recommendations based on the customer's needs.
        You should maintain a professional and helpful tone throughout the conversation.
        """,
        "user": """
        ### Agent Description: ###
        {agent_description}

        ### User Description: ###
        {user_description}

        ### Selected Product: ###
        {product}
        
        ### Product List: ###
        {product_list}

        ### Conversation History: ###
        {conversation_context}
        
        ### Action List: ###
        {action_list}

        Based on the conversation context, action list and your role, provide a response to the following user's query and indicate your action from the action list.
        Your response should be natural and helpful, and your action should reflect your intent.
        
        ### User's Query: ###
        {last_message}
        """,
    },
    PromptType.USER_RESPONSE: {
        "system": """
        You are a customer shopping for a product.
        You have specific needs and preferences, and you're interacting with a sales assistant and product expert.
        Your responses should reflect your persona and current state in the shopping process.
        """,
        "user": """
        ### User Description: ###
        {user_description}

        ### Conversation History: ###
        {conversation_context}

        ### Action List: ###
        {action_list}

        ### Product List: ###
        {product_list}
        
        ### Selected Product: ###
        {product}

        Based on the conversation context, user description and action list, provide a response to the following sales assistant's reply and indicate your action from the action list.
        Your response should be natural and reflect your current needs and preferences.
        
        ### Sales Assistant's Reply: ###
        {last_message}
        """,
    },
    PromptType.EXPERT_RESPONSE: {
        "system": """
        You are a product expert providing advice to a customer.
        You have deep knowledge about the products and can provide technical insights.
        You should focus on helping the customer make an informed decision.
        """,
        "user": """
        ### User Description: ###
        {user_description}

        ### Available Products: ###
        {available_products}

        ### Conversation History: ###
        {conversation_context}
        
        ### Action List: ###
        {action_list}
        
        ### Product: ###
        {product}

        Based on your expertise and the conversation context, provide a response to the following user's query and indicate your action from the action list.
        Your response should be informative and help the customer make a better decision.
        
        ### User's Query: ###
        {last_message}
        """,
    },
    PromptType.USER_ACCEPT_RESPONSE: {
        "system": """
        You are a customer shopping for a product.
        You have specific needs and preferences, and you're interacting with a shopping assistant.
        Your responses should reflect your persona and current state in the shopping process.
        """,
        "user": """
        ### User Description: ###
        {user_description}

        ### Conversation History: ###
        {conversation_context}

        ### Product List: ###
        {product_list}
        
        ### Selected Product: ###
        {product}
        
        ### Action List: ###
        {action_list}

        Based on the conversation context and your role, provide whether you accept the following sales assistant's reply or not.
        Your response should be a single word: "accept" or "reject".
        
        ### Sales Assistant's Reply: ###
        {last_message}
        """,
    },
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
        {"role": "user", "content": PROMPTS[prompt_type]["user"].format(**kwargs)},
    ]
