import os
import pandas as pd
from typing import Annotated, Sequence, TypeVar, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import Graph, START, END
from langchain_core.messages.chat import ChatMessage
import google.generativeai as genai

from IPython.display import Image, display


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

products_df = pd.read_csv('data.csv')


class InquiryCategory(str, Enum):
    PRODUCT = "product"
    SKINCARE_ADVICE = "skincare_advice"
    ORDER = "order"
    RETURNS = "returns"
    GENERAL = "general"
    UNKNOWN = "unknown"


class CustomerContext(BaseModel):
    customer_id: str
    history: list[dict] = Field(default_factory=list)
    current_category: InquiryCategory = InquiryCategory.UNKNOWN
    attempts: int = 0
    max_attempts: int = 3


class State(BaseModel):
    messages: list[ChatMessage]
    context: CustomerContext


def get_product_info(query: str) -> str:
    """Search product database and return relevant information"""
    relevant_products = products_df[
        products_df.apply(lambda row: any(query.lower() in str(value).lower() 
                                        for value in row), axis=1)
    ]
    
    if len(relevant_products) == 0:
        return "No matching products found."
    
    product_info = []
    for _, product in relevant_products.iterrows():
        info = f"""
                Product: {product['name']}
                Category: {product['category']}
                Price: ${product['price']}
                Description: {product['description']}
                Key Ingredients: {product['ingredients']}
                Suitable for: {product['skin_type']} skin types
                """
        product_info.append(info)
    
    return "\n".join(product_info)


def classify_inquiry(content: str) -> InquiryCategory:
    """Classify customer inquiry using Gemini API"""
    prompt = """You are a skincare company's customer service classifier. 
    Categorize the following inquiry into one of these categories: 
    - product (questions about specific products, ingredients, or recommendations)
    - skincare_advice (general skincare questions or concerns)
    - order (order status, shipping, payment issues)
    - returns (returns, refunds, product issues)
    - general (other general inquiries)
    
    Respond with just the category name in lowercase.
    
    Inquiry: {content}
    """.format(content=content)
    
    response = model.generate_content(prompt)
    category = response.text.strip().lower()
    return InquiryCategory(category)


def generate_response(category: InquiryCategory, inquiry: str, history: list[dict]) -> str:
    """Generate response using Gemini API based on category and context"""
    history_context = "\n".join([f"Previous interaction: {h['message']}" for h in history[-3:]])
    product_context = get_product_info(inquiry)
    
    prompt = """You are a knowledgeable and helpful customer service agent for a premium skincare company.
    Your name is Emma and you specialize in skincare advice and product recommendations.
    
    Previous interactions:
    {history}
    
    Relevant product information:
    {product_info}
    
    Customer inquiry ({category}): {inquiry}
    
    Guidelines:
    - Be friendly and professional
    - Provide specific product recommendations when relevant
    - Include product prices and key ingredients
    - Offer skincare advice based on customer concerns
    - For order/returns issues, explain the process clearly
    - If you don't have specific information, provide general guidance
    
    Provide a helpful and empathetic response.
    """.format(
        category=category.value,
        history=history_context,
        product_info=product_context,
        inquiry=inquiry
    )
    
    response = model.generate_content(prompt)
    return response.text


# Node functions
async def start_node(state: Dict) -> Dict:
    """Initial node that receives customer inquiry"""
    print("\nProcessing new inquiry...")
    return state


async def classification_node(state: Dict) -> Dict:
    """Classify the customer inquiry"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    category = classify_inquiry(latest_message)
    context["current_category"] = category.value
    print(f"Classified as: {category.value}")
    
    return {"messages": messages, "context": context, "result": category.value}


async def product_node(state: Dict) -> Dict:
    """Handle product-related inquiries"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    response = generate_response(InquiryCategory.PRODUCT, latest_message, context["history"])
    messages.append(ChatMessage(role="assistant", content=response))
    
    return {"messages": messages, "context": context}


async def skincare_advice_node(state: Dict) -> Dict:
    """Handle skincare advice inquiries"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    response = generate_response(InquiryCategory.SKINCARE_ADVICE, latest_message, context["history"])
    messages.append(ChatMessage(role="assistant", content=response))
    
    return {"messages": messages, "context": context}


async def order_node(state: Dict) -> Dict:
    """Handle order-related inquiries"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    response = generate_response(InquiryCategory.ORDER, latest_message, context["history"])
    messages.append(ChatMessage(role="assistant", content=response))
    
    return {"messages": messages, "context": context}


async def returns_node(state: Dict) -> Dict:
    """Handle returns and refunds inquiries"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    response = generate_response(InquiryCategory.RETURNS, latest_message, context["history"])
    messages.append(ChatMessage(role="assistant", content=response))
    
    return {"messages": messages, "context": context}


async def general_node(state: Dict) -> Dict:
    """Handle general inquiries"""
    messages = state["messages"]
    context = state["context"]
    latest_message = messages[-1].content
    
    response = generate_response(InquiryCategory.GENERAL, latest_message, context["history"])
    messages.append(ChatMessage(role="assistant", content=response))
    
    return {"messages": messages, "context": context}


async def end_node(state: Dict) -> Dict:
    """Final node that sends response back to customer"""
    messages = state["messages"]
    context = state["context"]
    
    # Update customer history
    context["history"].append({
        "timestamp": datetime.now().isoformat(),
        "category": context["current_category"],
        "message": messages[-2].content,
        "response": messages[-1].content
    })
    
    print(f"\nResponse: {messages[-1].content}")
    return {"messages": messages, "context": context}


# Create the graph
workflow = Graph()

# Add nodes
workflow.add_node("start", start_node)
workflow.add_node("classify", classification_node)
workflow.add_node("product", product_node)
workflow.add_node("skincare_advice", skincare_advice_node)
workflow.add_node("order", order_node)
workflow.add_node("returns", returns_node)
workflow.add_node("general", general_node)
workflow.add_node("end", end_node)

# Add edges
workflow.add_edge(START, "start")
workflow.add_edge("start", "classify")

# Define conditional edges
workflow.add_conditional_edges(
    "classify",
    lambda x: x["result"],
    {
        "product": "product",
        "skincare_advice": "skincare_advice",
        "order": "order",
        "returns": "returns",
        "general": "general"
    }
)

workflow.add_edge("product", "end")
workflow.add_edge("skincare_advice", "end")
workflow.add_edge("order", "end")
workflow.add_edge("returns", "end")
workflow.add_edge("general", "end")
workflow.add_edge("end", END)

# Compile the graph
app = workflow.compile()


async def process_inquiry(customer_id: str, message: str):
    """Process a customer inquiry through the workflow"""
    # Initialize state
    initial_state = {
        "messages": [ChatMessage(role="user", content=message)],
        "context": {
            "customer_id": customer_id,
            "history": [],
            "current_category": InquiryCategory.UNKNOWN.value,
            "attempts": 0,
            "max_attempts": 3
        }
    }
    
    # Run the workflow
    final_state = await app.ainvoke(initial_state)
    return final_state


if __name__ == "__main__":
    import asyncio
    
    async def main():
        customer_id = "12345"
        
        print("\nWelcome to our Skincare Customer Service!")
        print("How can I help you today? (Type 'quit' to exit)")
        
        while True:
            print("\nYour inquiry:")
            inquiry = input()
            
            if inquiry.lower() == 'quit':
                print("\nThank you for contacting our customer service. Have a great day!")
                break
                
            result = await process_inquiry(customer_id, inquiry)
            
    asyncio.run(main())