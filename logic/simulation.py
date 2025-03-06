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
        

class Cart:
    def __init__(self):
        self.items: Dict[str, int] = {}  # Dictionary of product names and their quantities
        
    def add_item(self, product: str, quantity: int = 1):
        if product in self.items:
            self.items[product] += quantity
        else:
            self.items[product] = quantity
            
    def remove_item(self, product: str, quantity: int = 1):
        if product in self.items:
            self.items[product] = max(0, self.items[product] - quantity)
            if self.items[product] == 0:
                del self.items[product]
                
    def get_items(self) -> Dict[str, int]:
        return self.items.copy()

class Simulation:
    def __init__(self):
        self.user: Optional[User] = None
        self.agent: Optional[Agent] = None
        self.cart = Cart()
        
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
        
    def initialize_user(self, persona: Persona, preference: Preference, goal: Goal):
        """Initialize a user with specific traits"""
        self.user = User(persona, preference, goal)
        
    def initialize_agent(self, strategy: Strategy):
        """Initialize an agent with specific strategy"""
        self.agent = Agent(strategy)
        
    def run(self):
        if not self.user:
            raise ValueError("User not initialized. Please call initialize_user first.")
        if not self.agent:
            raise ValueError("Agent not initialized. Please call initialize_agent first.")
        # TODO: Implement main simulation loop
        pass
