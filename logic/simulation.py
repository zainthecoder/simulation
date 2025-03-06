from enum import Enum
from typing import List, Dict, Optional
import random

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

# TODO: Send prompt to LLM and get response
# questions = llm.generate(user_traits_prompt)

# For now, print the prompt that would be sent
print("\nPrompt that would be sent to LLM:")
print(user_traits_prompt)

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Get question from first prompt
def get_question_from_prompt(prompt):
    # For demonstration, using a simple question about user traits
    # In practice, this would be more sophisticated based on the prompt
    question = "What specific needs does this user have based on their traits?"
    return question

# Build second prompt with agent strategy and question
def build_answer_prompt(question, agent_strategy):
    answer_prompt = f"""
Agent Strategy: {agent_strategy}

Question from analysis: {question}

Based on the agent's strategy and the question above, provide a detailed response
that addresses the user's specific needs and aligns with the agent's approach.
"""
    return answer_prompt

# Build third prompt for user decision
def build_user_decision_prompt(agent_response):
    user_prompt = f"""
Based on the previous interaction:

Agent's Response: {agent_response}

As a user with the following traits:
- Persona: {sim.user.persona}
- Preference: {sim.user.preference}
- Goal: {sim.user.goal}

What would you like to do?
[UserSeek] Seek expert advice
[UserDecide] Choose an item to buy

Please respond with your choice and brief explanation.
"""
    return user_prompt

# Build fourth prompt based on user's decision
def build_fourth_prompt(user_decision, user_explanation, agent_response):
    if "UserSeek" in user_decision:
        expert_prompt = f"""
As an expert advisor, considering:
- User's explanation: {user_explanation}
- Previous agent response: {agent_response}
- Available products: {[p.name for p in current_products]}

Provide detailed expert advice to help the user make an informed decision.
"""
        return expert_prompt
    else:
        agent_prompt = f"""
As a sales agent, considering:
- User's choice to purchase: {user_explanation}
- Previous interaction: {agent_response}
- Available products: {[p.name for p in current_products]}

Recommend the most suitable product and explain why.
"""
        return agent_prompt

# Get answer using RoBERTa model
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end+1]
        )
    )
    return answer

# Execute the conversation flow
first_question = get_question_from_prompt(user_traits_prompt)
print("\nQuestion generated from first prompt:")
print(first_question)

agent_strategy = sim.agent.strategy if hasattr(sim.agent, 'strategy') else "Default strategy"
second_prompt = build_answer_prompt(first_question, agent_strategy)
agent_response = get_answer(first_question, second_prompt)
print("\nAgent Response:")
print(agent_response)

# Third prompt - User decision
user_prompt = build_user_decision_prompt(agent_response)
user_response = get_answer(first_question, user_prompt)  # Using first_question as context
print("\nUser Decision:")
print(user_response)

# Fourth prompt - Expert/Agent response
fourth_prompt = build_fourth_prompt(user_response, user_response, agent_response)
final_response = get_answer(first_question, fourth_prompt)
print("\nFinal Response (Expert/Agent):")
print(final_response)


