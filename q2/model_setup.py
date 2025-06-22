"""
Local Model Setup for Hallucination Detection
This file shows how to connect local models with our knowledge base
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

class LocalModelHandler:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize local model for question answering
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        
    def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create a simple text generation pipeline
            self.qa_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=100,
                temperature=0.7
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def ask_question(self, question):
        """
        Ask a question to the local model
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: Model's response
        """
        if not self.qa_pipeline:
            return "Model not loaded"
        
        try:
            # Format the question for the model
            prompt = f"Question: {question}\nAnswer:"
            
            # Generate response
            response = self.qa_pipeline(prompt, max_length=len(prompt.split()) + 20)
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract just the answer part
            answer = generated_text.split("Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {e}"

def load_knowledge_base(kb_file="kb.json"):
    """
    Load our knowledge base
    
    Args:
        kb_file (str): Path to knowledge base file
        
    Returns:
        dict: Knowledge base data
    """
    try:
        with open(kb_file, 'r') as f:
            kb_data = json.load(f)
        print(f"✅ Knowledge base loaded: {len(kb_data.get('questions', []))} questions")
        return kb_data
    except FileNotFoundError:
        print(f"❌ Knowledge base file {kb_file} not found")
        return {"questions": []}

def test_model_with_kb():
    """
    Test how the model works with our knowledge base
    """
    print("🧪 Testing Model with Knowledge Base")
    print("=" * 50)
    
    # Load knowledge base
    kb_data = load_knowledge_base()
    
    # Initialize model
    model_handler = LocalModelHandler()
    if not model_handler.load_model():
        print("❌ Failed to load model")
        return
    
    # Test with a sample question
    sample_question = "What is the capital of France?"
    print(f"\n📝 Question: {sample_question}")
    
    # Get model's answer
    model_answer = model_handler.ask_question(sample_question)
    print(f"🤖 Model Answer: {model_answer}")
    
    # Check against KB (if we had this question)
    kb_answer = "Paris"  # Assuming this is in our KB
    print(f"📚 KB Answer: {kb_answer}")
    
    # Simple comparison
    if kb_answer.lower() in model_answer.lower():
        print("✅ Model answer matches KB!")
    else:
        print("❌ Model answer differs from KB - potential hallucination")

if __name__ == "__main__":
    test_model_with_kb() 