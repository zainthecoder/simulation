from enum import Enum
from typing import List, Dict, Optional
import random
import json
import logging
from jsonformer import Jsonformer
from model_config import initialize_model, get_bnb_config

# Load access token
with open('token.json', 'r') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

class Persona(Enum):
    PARENT = "parent_shopping_for_kids_phone"
    COLLEGE_STUDENT = "college_student_basic_phone"
    SOCIAL_MEDIA = "social_media_enthusiast"

class Preference(Enum):
    PRICE_ORIENTED = "price_oriented"
    RATING_ORIENTED = "rating_oriented"
    NONE = "none"

class Goal(Enum):
    INFORMATION_SEEKING = "information_seeking"
    DECISION_MAKING = "decision_making"

class Strategy(Enum):
    REVENUE_ORIENTED = "revenue_oriented"
    PRODUCT_ORIENTED = "product_oriented"
    CUSTOMER_ORIENTED = "customer_oriented"

class User:
    def __init__(self, persona: Persona, preference: Preference, goal: Goal):
        self.persona = persona
        self.preference = preference
        self.goal = goal
        
    def get_user_description(self) -> str:
        """Generate a natural language description of the user"""
        descriptions = {
            Persona.PARENT: "A parent shopping for their kid's first smartphone",
            Persona.COLLEGE_STUDENT: "A college student looking for a reliable phone with basic features",
            Persona.SOCIAL_MEDIA: "A social media enthusiast who needs a phone for photos and videos"
        }
        
        preference_desc = {
            Preference.PRICE_ORIENTED: "who is primarily concerned about price",
            Preference.RATING_ORIENTED: "who values high ratings and reviews",
            Preference.NONE: "who doesn't have strong preferences"
        }
        
        goal_desc = {
            Goal.INFORMATION_SEEKING: "and is currently gathering information",
            Goal.DECISION_MAKING: "and is ready to make a purchase decision"
        }
        
        return f"{descriptions[self.persona]}, {preference_desc[self.preference]}, {goal_desc[self.goal]}"

class Agent:
    def __init__(self, strategy: Strategy):
        self.name = "Shopping Assistant"
        self.strategy = strategy
        
    def get_agent_description(self) -> str:
        """Generate a natural language description of the agent"""
        strategy_desc = {
            Strategy.REVENUE_ORIENTED: "focused on maximizing sales and revenue",
            Strategy.PRODUCT_ORIENTED: "focused on promoting high-quality products",
            Strategy.CUSTOMER_ORIENTED: "focused on customer satisfaction and needs"
        }
        return f"A shopping assistant {strategy_desc[self.strategy]}"
        

class Product:
    def __init__(self, name: str):
        self.name = name
        
    def get_description(self) -> str:
        return self.name

class Simulation:
    def __init__(self):
        self.user: Optional[User] = None
        self.agent: Optional[Agent] = None
        
        # Initialize model
        logging.info("Starting model initialization...")
        bnb_config = get_bnb_config()
        
        self.model, self.tokenizer = initialize_model(
            access_token=access_token,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            bnb_config=bnb_config
        )
        logging.info("Model loaded successfully")
        
        # Create all possible products
        self.all_products = {
            "iPhone 13": Product("iPhone 13"),
            "Samsung Galaxy S21": Product("Samsung Galaxy S21"), 
            "Google Pixel 6": Product("Google Pixel 6"),
            "OnePlus 9": Product("OnePlus 9"),
            "Motorola Edge": Product("Motorola Edge"),
            "Xiaomi Redmi Note 10": Product("Xiaomi Redmi Note 10")
        }
        
        # Product reviews
        self.product_reviews = {
            "iPhone 13": "Great performance and camera quality, but battery life could be better. Premium build quality though expensive.",
            "Samsung Galaxy S21": "Excellent all-around phone with great display and features. One of the best Android phones available.",
            "Google Pixel 6": "Outstanding camera system and clean Android experience, but higher price point than previous models.",
            "OnePlus 9": "Fast charging and smooth performance at a reasonable price point. Good value flagship phone.",
            "Motorola Edge": "Solid mid-range option with good battery life and clean Android interface. Great for everyday use.",
            "Xiaomi Redmi Note 10": "Impressive features for the price point. Best budget phone with good camera and battery life."
        }
        
        # List of product lists
        self.product_lists = [
            [self.all_products["iPhone 13"], self.all_products["Samsung Galaxy S21"], self.all_products["Motorola Edge"]],
            [self.all_products["Samsung Galaxy S21"], self.all_products["Xiaomi Redmi Note 10"]],
            [self.all_products["Samsung Galaxy S21"], self.all_products["iPhone 13"], self.all_products["OnePlus 9"]]
        ]
        
        # Current product list (will be set when simulation starts)
        self.current_product_list: Optional[List[Product]] = None
        
    def initialize_random_user(self):
        """Initialize a user with random traits"""
        persona = random.choice(list(Persona))
        preference = random.choice(list(Preference))
        goal = random.choice(list(Goal))
        
        self.user = User(persona, preference, goal)
        return self.user
        
    def initialize_random_agent(self):
        """Initialize an agent with random strategy"""
        strategy = random.choice(list(Strategy))
        self.agent = Agent(strategy)
        return self.agent
        
    def select_random_product_list(self):
        """Select a random product list for the simulation"""
        self.current_product_list = random.choice(self.product_lists)
        return self.current_product_list
        
    def get_product_review(self, product: Product) -> str:
        """Get the review for a specific product"""
        return self.product_reviews.get(product.name, "No review available.")
        
    def run(self):
        if not self.user:
            raise ValueError("User not initialized. Please call initialize_user first.")
        if not self.agent:
            raise ValueError("Agent not initialized. Please call initialize_agent first.")
        if not self.current_product_list:
            raise ValueError("Product list not selected. Please call select_random_product_list first.")
        # TODO: Implement main simulation loop
        pass
    
    def run_prompting(self, tokenizer, model, messages, json_schema):
        """
        Run the prompting process using the provided model and tokenizer.
        
        Args:
            tokenizer: The tokenizer to use for processing the messages
            model: The language model to use
            messages: List of messages to process
            json_schema: The JSON schema for formatting the response
            
        Returns:
            tuple: (prompt, response) where prompt is the processed input and
                  response is the model's structured output
        """
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        jsonformer = Jsonformer(
            model, 
            tokenizer, 
            json_schema, 
            prompt, 
            max_number_tokens=2000,
            max_array_length=2000,
            max_string_token_length=2000
        )
        
        response = jsonformer()
        return (prompt, response)


sim = Simulation()

# Initialize user and agent
sim.initialize_random_user()
sim.initialize_random_agent()

# Select a random product list
current_products = sim.select_random_product_list()

# Print current products and their reviews
for product in current_products:
    review = sim.get_product_review(product)
    print(f"{product.name}: {review}")

# print user description
print(sim.user.get_user_description())

# Build initial prompt with user traits and product context
user_traits_prompt = f"""
User traits:
- Persona: {sim.user.persona}
- Preference: {sim.user.preference} 
- Goal: {sim.user.goal}

Available products:
{[p.name for p in current_products]}

Based on the user traits and available products, generate 3-4 general questions to ask about the recommended items.
Questions should help understand the user's needs better.
"""

