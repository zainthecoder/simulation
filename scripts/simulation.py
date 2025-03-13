from enum import Enum
from typing import List, Dict, Optional, Tuple
import random
import json
import logging
from jsonformer import Jsonformer
from model_config import initialize_model, get_bnb_config
from prompts import PromptType, get_prompt
import pprint
from dataclasses import dataclass
from datetime import datetime

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

class UserAction(Enum):
    ASK = "ask_general_questions"
    SEEK = "seek_expert_advice"
    DECIDE = "choose_item"
    ACCEPT = "accept_response"
    REJECT = "reject_response"

class AgentAction(Enum):
    CLARIFICATION = "ask_clarification"
    ANSWER = "answer_question"
    RECOMMEND = "make_recommendation"
    DISAGREE = "disagree_with_expert"
    AGREE = "agree_with_expert"
    REFINE = "refine_recommendation"
    WARRANTY = "discuss_warranty"

class ExpertAction(Enum):
    DISAGREE = "disagree_with_agent"
    AGREE = "agree_with_agent"
    SUGGEST = "suggest_new_product"

class Role(Enum):
    USER = "user"
    AGENT = "agent"
    EXPERT = "expert"

@dataclass
class Message:
    role: Role
    content: str
    action: Optional[Enum] = None
    timestamp: datetime = datetime.now()

class ConversationHistory:
    def __init__(self):
        self.messages: List[Message] = []
        
    def add_message(self, role: Role, content: str, action: Optional[Enum] = None):
        """Add a new message to the conversation history"""
        self.messages.append(Message(role=role, content=content, action=action))
        
    def get_context(self) -> str:
        """Get the full conversation context as a formatted string"""
        context = []
        for msg in self.messages:
            role_prefix = f"{msg.role.value.upper()}: "
            context.append(f"{role_prefix}{msg.content}")
        return "\n".join(context)
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None

class Strategy(Enum):
    REVENUE_ORIENTED = "revenue_oriented"
    PRODUCT_ORIENTED = "product_oriented"
    CUSTOMER_ORIENTED = "customer_oriented"

class User:
    def __init__(self, persona: Persona, preference: Preference, goal: Goal):
        self.persona = persona
        self.preference = preference
        self.goal = goal
        self.current_action: Optional[UserAction] = None
        
    def set_action(self, action: UserAction):
        """Set the current action for the user"""
        self.current_action = action
        
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
        
        action_desc = {
            UserAction.ASK: "asking general questions about products",
            UserAction.SEEK: "seeking expert advice",
            UserAction.DECIDE: "ready to make a purchase decision",
            UserAction.ACCEPT: "accepting the previous response",
            UserAction.REJECT: "rejecting the previous response"
        }
        
        base_desc = f"{descriptions[self.persona]}, {preference_desc[self.preference]}, {goal_desc[self.goal]}"
        if self.current_action:
            base_desc += f", {action_desc[self.current_action]}"
            
        return base_desc

    def can_perform_action(self, action: UserAction) -> bool:
        """Check if the user can perform the given action based on their current state"""
        if action in [UserAction.ACCEPT, UserAction.REJECT]:
            return True  # Can always accept/reject responses
        elif action == UserAction.ASK:
            return True  # Can always ask questions
        elif action == UserAction.SEEK:
            return self.goal == Goal.INFORMATION_SEEKING  # Can seek advice while gathering information
        elif action == UserAction.DECIDE:
            return self.goal == Goal.DECISION_MAKING  # Can only decide when in decision-making mode
        return False

class Agent:
    def __init__(self, strategy: Strategy):
        self.name = "Shopping Assistant"
        self.strategy = strategy
        self.current_action: Optional[AgentAction] = None
        
    def set_action(self, action: AgentAction):
        """Set the current action for the agent"""
        self.current_action = action
        
    def get_agent_description(self) -> str:
        """Generate a natural language description of the agent"""
        strategy_desc = {
            Strategy.REVENUE_ORIENTED: "focused on maximizing sales and revenue",
            Strategy.PRODUCT_ORIENTED: "focused on promoting high-quality products",
            Strategy.CUSTOMER_ORIENTED: "focused on customer satisfaction and needs"
        }
        
        action_desc = {
            AgentAction.CLARIFICATION: "asking for clarification",
            AgentAction.ANSWER: "answering questions",
            AgentAction.RECOMMEND: "making recommendations",
            AgentAction.DISAGREE: "disagreeing with expert advice",
            AgentAction.AGREE: "agreeing with expert advice",
            AgentAction.REFINE: "refining recommendations",
            AgentAction.WARRANTY: "discussing warranty options"
        }
        
        base_desc = f"A shopping assistant {strategy_desc[self.strategy]}"
        if self.current_action:
            base_desc += f", {action_desc[self.current_action]}"
            
        return base_desc

class Expert:
    def __init__(self):
        self.name = "Product Expert"
        self.current_action: Optional[ExpertAction] = None
        
    def set_action(self, action: ExpertAction):
        """Set the current action for the expert"""
        self.current_action = action
        
    def get_expert_description(self) -> str:
        """Generate a natural language description of the expert"""
        action_desc = {
            ExpertAction.DISAGREE: "disagreeing with the agent's advice",
            ExpertAction.AGREE: "agreeing with the agent's advice",
            ExpertAction.SUGGEST: "suggesting alternative products"
        }
        
        base_desc = "A product expert"
        if self.current_action:
            base_desc += f", {action_desc[self.current_action]}"
            
        return base_desc

class Product:
    def __init__(self, name: str):
        self.name = name
        
    def get_description(self) -> str:
        return self.name

class Simulation:
    def __init__(self):
        self.user: Optional[User] = None
        self.agent: Optional[Agent] = None
        self.expert: Optional[Expert] = None
        self.conversation_history = ConversationHistory()
        
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
        
    def generate_agent_response(self) -> Tuple[AgentAction, str]:
        """Generate the agent's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.AGENT_RESPONSE,
            agent_description=self.agent.get_agent_description(),
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            available_products=[p.name for p in self.current_product_list]
        )
        
        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in AgentAction]
                },
                "response": {"type": "string"}
            },
            "required": ["action", "response"]
        }
        
        response = self.generate_structured_response(prompt, json_schema)
        action = AgentAction(response["action"])
        self.agent.set_action(action)
        return action, response["response"]

    def generate_user_response(self) -> Tuple[UserAction, str]:
        """Generate the user's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.USER_RESPONSE,
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            last_message=self.conversation_history.get_last_message().content if self.conversation_history.get_last_message() else None
        )
        
        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in UserAction]
                },
                "response": {"type": "string"}
            },
            "required": ["action", "response"]
        }
        
        response = self.generate_structured_response(prompt, json_schema)
        action = UserAction(response["action"])
        if self.user.can_perform_action(action):
            self.user.set_action(action)
            return action, response["response"]
        else:
            # If action is not allowed, default to asking a question
            self.user.set_action(UserAction.ASK)
            return UserAction.ASK, "I have a question about the products."

    def generate_expert_response(self) -> Tuple[ExpertAction, str]:
        """Generate the expert's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.EXPERT_RESPONSE,
            expert_description=self.expert.get_expert_description(),
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            available_products=[p.name for p in self.current_product_list]
        )
        
        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in ExpertAction]
                },
                "response": {"type": "string"}
            },
            "required": ["action", "response"]
        }
        
        response = self.generate_structured_response(prompt, json_schema)
        action = ExpertAction(response["action"])
        self.expert.set_action(action)
        return action, response["response"]

    def run(self):
        """Run the main simulation loop"""
        if not self.user:
            raise ValueError("User not initialized. Please call initialize_user first.")
        if not self.agent:
            raise ValueError("Agent not initialized. Please call initialize_agent first.")
        if not self.current_product_list:
            raise ValueError("Product list not selected. Please call select_random_product_list first.")
            
        print("\nStarting conversation...")
        print(f"User: {self.user.get_user_description()}")
        print(f"Agent: {self.agent.get_agent_description()}")
        
        while True:
            # Step 1: Agent responds
            agent_action, agent_response = self.generate_agent_response()
            self.conversation_history.add_message(Role.AGENT, agent_response, agent_action)
            print(f"\nAgent: {agent_response}")
            
            # Step 2: User reacts
            user_action, user_response = self.generate_user_response()
            self.conversation_history.add_message(Role.USER, user_response, user_action)
            print(f"\nUser: {user_response}")
            
            # Check if user decided to buy
            if user_action == UserAction.DECIDE:
                print("\nUser has decided to make a purchase. Conversation ended.")
                break
                
            # Step 3: Conditional flow based on user acceptance
            if user_action == UserAction.REJECT:
                # If user rejected agent's response, continue with agent
                continue
            elif user_action == UserAction.ACCEPT:
                # If user accepted, check next action
                next_user_action, next_user_response = self.generate_user_response()
                self.conversation_history.add_message(Role.USER, next_user_response, next_user_action)
                print(f"\nUser: {next_user_response}")
                
                if next_user_action == UserAction.SEEK:
                    # Step 4: Expert interaction loop
                    while True:
                        # Expert responds
                        expert_action, expert_response = self.generate_expert_response()
                        self.conversation_history.add_message(Role.EXPERT, expert_response, expert_action)
                        print(f"\nExpert: {expert_response}")
                        
                        # User reacts to expert
                        expert_user_action, expert_user_response = self.generate_user_response()
                        self.conversation_history.add_message(Role.USER, expert_user_response, expert_user_action)
                        print(f"\nUser: {expert_user_response}")
                        
                        if expert_user_action == UserAction.ACCEPT:
                            # Step 6: Agent final response
                            agent_action, agent_response = self.generate_agent_response()
                            self.conversation_history.add_message(Role.AGENT, agent_response, agent_action)
                            print(f"\nAgent: {agent_response}")
                            break  # Exit expert loop when user accepts
                        elif expert_user_action == UserAction.REJECT:
                            # If user rejects expert advice, continue expert loop
                            continue
                        elif expert_user_action == UserAction.DECIDE:
                            print("\nUser has decided to make a purchase. Conversation ended.")
                            return  # Exit entire simulation
                        elif expert_user_action == UserAction.ASK:
                            # If user asks a question, break expert loop and return to agent
                            break
                elif next_user_action == UserAction.DECIDE:
                    print("\nUser has decided to make a purchase. Conversation ended.")
                    break
                elif next_user_action == UserAction.ASK:
                    # If user asks a question, continue with agent
                    continue

    def generate_structured_response(self, prompt: str, json_schema: dict) -> dict:
        """
        Generate a structured response from the model using JSONformer.
        
        Args:
            prompt: The input prompt
            json_schema: The JSON schema defining the expected response structure
            
        Returns:
            dict: Structured response from the model
        """
        logging.info("Generating structured response...")
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Initialize JSONformer
        jsonformer = Jsonformer(
            self.model,
            self.tokenizer,
            json_schema,
            formatted_prompt,
            max_number_tokens=2000,
            max_array_length=2000,
            max_string_token_length=2000
        )
        
        # Generate response
        response = jsonformer()
        logging.info("Successfully generated structured response")
        return response

    def generate_initial_questions(self) -> dict:
        """Generate initial questions based on user traits and available products"""
        prompt = get_prompt(
            PromptType.INITIAL_QUESTIONS,
            persona=self.user.persona.value,
            preference=self.user.preference.value,
            goal=self.user.goal.value,
            products=[p.name for p in self.current_product_list]
        )
        print("\nGenerated Prompt:")
        pprint.pprint(prompt)
        
        # Define JSON schema for questions
        json_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "product": {"type": "string"}
            }
        }
        
        return self.generate_structured_response(prompt, json_schema)

    # def generate_product_recommendation(self) -> str:
    #     """Generate product recommendation based on user traits and reviews"""
    #     product_reviews = "\n".join([
    #         f"{p.name}: {self.product_reviews[p.name]}"
    #         for p in self.current_product_list
    #     ])
        
    #     prompt = get_prompt(
    #         PromptType.PRODUCT_RECOMMENDATION,
    #         persona=self.user.persona.value,
    #         preference=self.user.preference.value,
    #         goal=self.user.goal.value,
    #         product_reviews=product_reviews
    #     )
    #     return self._generate_response(prompt)

    def _generate_response(self, prompt: str) -> str:
        """Generate response from the model for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Example usage
if __name__ == "__main__":
    sim = Simulation()
    
    # Initialize user, agent, and expert
    sim.initialize_random_user()
    sim.initialize_random_agent()
    
    # Select a random product list
    current_products = sim.select_random_product_list()
    
    # Generate initial questions
    print("\nGenerating initial questions...")
    questions_response = sim.generate_initial_questions()
    
    print(questions_response)
    
    # Now i have 


