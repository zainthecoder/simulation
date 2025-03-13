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
with open("token.json", "r") as f:
    token_data = json.load(f)
    access_token = token_data["access_token"]


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


class UserAcceptAction(Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class AgentAction(Enum):
    CLARIFICATION = "ask_clarification"
    ANSWER = "answer_question"
    RECOMMEND = "make_recommendation"
    DISAGREE = "disagree_with_agent"
    AGREE = "agree_with_agent"
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

    def save_to_file(self, filename: str = None):
        """Save the conversation history to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        conversation_data = {
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "action": msg.action.value if msg.action else None,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in self.messages
            ]
        }

        with open(filename, "w") as f:
            json.dump(conversation_data, f, indent=2)

        return filename


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
            Persona.SOCIAL_MEDIA: "A social media enthusiast who needs a phone for photos and videos",
        }

        preference_desc = {
            Preference.PRICE_ORIENTED: "who is primarily concerned about price",
            Preference.RATING_ORIENTED: "who values high ratings and reviews",
            Preference.NONE: "who doesn't have strong preferences",
        }

        goal_desc = {
            Goal.INFORMATION_SEEKING: "and is currently gathering information",
            Goal.DECISION_MAKING: "and is ready to make a purchase decision",
        }

        action_desc = {
            UserAction.ASK: "asking general questions about products",
            UserAction.SEEK: "seeking expert advice",
            UserAction.DECIDE: "ready to make a purchase decision",
            UserAcceptAction.ACCEPT: "accepting the previous response",
            UserAcceptAction.REJECT: "rejecting the previous response",
        }

        base_desc = f"{descriptions[self.persona]}, {preference_desc[self.preference]}, {goal_desc[self.goal]}"
        if self.current_action:
            base_desc += f", {action_desc[self.current_action]}"

        return base_desc

    def can_perform_action(self, action: UserAction) -> bool:
        """Check if the user can perform the given action based on their current state"""
        if action in [UserAcceptAction.ACCEPT, UserAcceptAction.REJECT]:
            return True  # Can always accept/reject responses
        elif action == UserAction.ASK:
            return True  # Can always ask questions
        elif action == UserAction.SEEK:
            return (
                self.goal == Goal.INFORMATION_SEEKING
            )  # Can seek advice while gathering information
        elif action == UserAction.DECIDE:
            return (
                self.goal == Goal.DECISION_MAKING
            )  # Can only decide when in decision-making mode
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
            Strategy.CUSTOMER_ORIENTED: "focused on customer satisfaction and needs",
        }

        action_desc = {
            AgentAction.CLARIFICATION: "asking for clarification",
            AgentAction.ANSWER: "answering questions",
            AgentAction.RECOMMEND: "making recommendations",
            AgentAction.DISAGREE: "disagreeing with expert advice",
            AgentAction.AGREE: "agreeing with expert advice",
            AgentAction.REFINE: "refining recommendations",
            AgentAction.WARRANTY: "discussing warranty options",
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
            ExpertAction.SUGGEST: "suggesting alternative products",
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
            bnb_config=bnb_config,
        )
        logging.info("Model loaded successfully")

        # Create all possible products
        self.all_products = {
            "iPhone 13": Product("iPhone 13"),
            "Samsung Galaxy S21": Product("Samsung Galaxy S21"),
            "Google Pixel 6": Product("Google Pixel 6"),
            "OnePlus 9": Product("OnePlus 9"),
            "Motorola Edge": Product("Motorola Edge"),
            "Xiaomi Redmi Note 10": Product("Xiaomi Redmi Note 10"),
        }

        # Product reviews
        self.product_reviews = {
            "iPhone 13": "Great performance and camera quality, but battery life could be better. Premium build quality though expensive.",
            "Samsung Galaxy S21": "Excellent all-around phone with great display and features. One of the best Android phones available.",
            "Google Pixel 6": "Outstanding camera system and clean Android experience, but higher price point than previous models.",
            "OnePlus 9": "Fast charging and smooth performance at a reasonable price point. Good value flagship phone.",
            "Motorola Edge": "Solid mid-range option with good battery life and clean Android interface. Great for everyday use.",
            "Xiaomi Redmi Note 10": "Impressive features for the price point. Best budget phone with good camera and battery life.",
        }

        # List of product lists
        self.product_lists = [
            [
                self.all_products["iPhone 13"],
                self.all_products["Samsung Galaxy S21"],
                self.all_products["Motorola Edge"],
            ],
            [
                self.all_products["Samsung Galaxy S21"],
                self.all_products["Xiaomi Redmi Note 10"],
            ],
            [
                self.all_products["Samsung Galaxy S21"],
                self.all_products["iPhone 13"],
                self.all_products["OnePlus 9"],
            ],
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

    def generate_agent_response(
        self, expert_action: Optional[AgentAction] = None
    ) -> Tuple[AgentAction, str]:
        """Generate the agent's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.AGENT_RESPONSE,
            agent_description=self.agent.get_agent_description(),
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            available_products=[p.name for p in self.current_product_list],
        )
        print("\nAgent Prompt:")
        pprint.pprint(prompt)

        if expert_action:
            ActionList = [
                AgentAction.DISAGREE,
                AgentAction.AGREE,
                AgentAction.REFINE,
                AgentAction.WARRANTY,
            ]
        else:
            ActionList = [
                AgentAction.CLARIFICATION,
                AgentAction.ANSWER,
                AgentAction.RECOMMEND,
            ]

        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in ActionList],
                },
                "response": {"type": "string"},
            },
            "required": ["action", "response"],
        }

        response = self.generate_structured_response(prompt, json_schema)
        action = AgentAction(response["action"])
        self.agent.set_action(action)
        print("\nAgent Response:")
        pprint.pprint(response)
        print("\nAgent Action:")
        pprint.pprint(action)
        return action, response["response"]

    def generate_user_response(self) -> Tuple[UserAction, str]:
        """Generate the user's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.USER_RESPONSE,
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            last_message=(
                self.conversation_history.get_last_message().content
                if self.conversation_history.get_last_message()
                else None
            ),
        )
        print("\nUser Prompt:")
        pprint.pprint(prompt)

        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in UserAction],
                },
                "response": {"type": "string"},
            },
            "required": ["action", "response"],
        }

        response = self.generate_structured_response(prompt, json_schema)
        action = UserAction(response["action"])
        if self.user.can_perform_action(action):
            self.user.set_action(action)
            print("\nUser Response:")
            pprint.pprint(response)
            print("\nUser Action:")
            pprint.pprint(action)
            return action, response["response"]
        else:
            # If action is not allowed, default to asking a question
            self.user.set_action(UserAction.ASK)
            return UserAction.ASK, "I have a question about the products."

    def generate_user_accept_response(self) -> Tuple[UserAcceptAction, str]:
        """Generate accept/reject response from user"""
        prompt = get_prompt(
            PromptType.USER_ACCEPT_RESPONSE,
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            last_message=(
                self.conversation_history.get_last_message().content
                if self.conversation_history.get_last_message()
                else None
            ),
        )
        print("\nUser Accept Prompt:")
        pprint.pprint(prompt)

        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in UserAcceptAction],
                },
            },
            "required": ["action"],
        }

        response = self.generate_structured_response(prompt, json_schema)
        action = UserAcceptAction(response["action"])
        print("\nUser Accept Response:")
        pprint.pprint(response)
        print("\nUser Accept Action:")
        pprint.pprint(action)
        return action, response.get("response", "")

    def generate_expert_response(self) -> Tuple[ExpertAction, str]:
        """Generate the expert's response based on the conversation context"""
        prompt = get_prompt(
            PromptType.EXPERT_RESPONSE,
            expert_description=self.expert.get_expert_description(),
            user_description=self.user.get_user_description(),
            conversation_context=self.conversation_history.get_context(),
            available_products=[p.name for p in self.current_product_list],
        )
        print("\nExpert Prompt:")
        pprint.pprint(prompt)

        json_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in ExpertAction],
                },
                "response": {"type": "string"},
            },
            "required": ["action", "response"],
        }

        response = self.generate_structured_response(prompt, json_schema)
        action = ExpertAction(response["action"])
        self.expert.set_action(action)
        print("\nExpert Response:")
        pprint.pprint(response)
        print("\nExpert Action:")
        pprint.pprint(action)
        return action, response["response"]

    def initialize_expert(self):
        """Initialize the expert"""
        self.expert = Expert()
        return self.expert

    def run(self):
        """Run the main simulation loop"""
        if not self.user:
            raise ValueError("User not initialized. Please call initialize_user first.")
        if not self.agent:
            raise ValueError("Agent not initialized. Please call initialize_agent first.")
        if not self.current_product_list:
            raise ValueError("Product list not selected. Please call select_random_product_list first.")
        
        # Add expert initialization
        if not self.expert:
            self.initialize_expert()
            
        print("\nStarting conversation...")
        print(f"User: {self.user.get_user_description()}")
        print(f"Agent: {self.agent.get_agent_description()}")
        
        # Start with initial questions
        initial_response = self.generate_initial_questions()
        question = initial_response.get("question", "")
        product = initial_response.get("product", "")
        
        # Set agent's first response using the initial question
        first_response = f"Let me ask you about {product}: {question}"
        self.conversation_history.add_message(Role.AGENT, first_response, AgentAction.CLARIFICATION)
        print(f"\nAgent: {first_response}")
        
        expert_flag = False
        
        while True:
            # Step 1: Agent responds
            agent_action, agent_response = self.generate_agent_response(expert_flag)
            self.conversation_history.add_message(
                Role.AGENT, agent_response, agent_action
            )
            print(f"\nAgent: {agent_response}")

            # Step 2: User reacts
            accept_action, accept_response = self.generate_user_accept_response()
            self.conversation_history.add_message(
                Role.USER, accept_response, accept_action
            )
            print(f"\nUser: {accept_response}")

            if accept_action == UserAcceptAction.REJECT:
                continue

            elif accept_action == UserAcceptAction.ACCEPT:

                user_action, user_response = self.generate_user_response()
                self.conversation_history.add_message(
                    Role.USER, user_response, user_action
                )
                print(f"\nUser: {user_response}")

                # Check if user decided to buy
                if user_action == UserAction.DECIDE:
                    print("\nUser has decided to make a purchase. Conversation ended.")
                    break

                if user_action == UserAction.SEEK:
                    # Expert interaction loop
                    while True:

                        expert_flag = True

                        expert_action, expert_response = self.generate_expert_response()
                        self.conversation_history.add_message(
                            Role.EXPERT, expert_response, expert_action
                        )
                        print(f"\nExpert: {expert_response}")

                        user_action, user_response = (
                            self.generate_user_accept_response()
                        )
                        self.conversation_history.add_message(
                            Role.USER, user_response, user_action
                        )
                        print(f"\nUser: {user_response}")

                        if user_action == UserAcceptAction.REJECT:
                            continue

                        if user_action == UserAcceptAction.ACCEPT:
                            break

        # Save conversation history at the end
        saved_file = self.conversation_history.save_to_file()
        print(f"\nConversation history saved to: {saved_file}")

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
            prompt, tokenize=False, add_generation_prompt=True, return_tensors="pt"
        )

        # Initialize JSONformer
        jsonformer = Jsonformer(
            self.model,
            self.tokenizer,
            json_schema,
            formatted_prompt,
            max_number_tokens=2000,
            max_array_length=2000,
            max_string_token_length=2000,
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
            products=[p.name for p in self.current_product_list],
        )
        print("\nGenerated Prompt:")
        pprint.pprint(prompt)

        # Define JSON schema for questions
        json_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "product": {"type": "string"},
            },
        }

        return self.generate_structured_response(prompt, json_schema)

    def _generate_response(self, prompt: str) -> str:
        """Generate response from the model for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# Example usage
if __name__ == "__main__":
    sim = Simulation()
    
    # Initialize user, agent, and expert
    sim.initialize_random_user()
    sim.initialize_random_agent()
    sim.initialize_expert()
    
    # Select a random product list
    current_products = sim.select_random_product_list()
    
    # Run the simulation (initial questions now handled in run())
    sim.run()
