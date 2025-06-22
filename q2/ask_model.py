"""
Model Question Asker for Hallucination Detection
Asks questions to the model and coordinates validation
"""

import json
import time
import logging
from typing import List, Dict, Optional
from model_setup import LocalModelHandler
from validator import HallucinationValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run.log'),
        logging.StreamHandler()
    ]
)

class ModelQuestionAsker:
    def __init__(self, kb_file: str = "kb.json"):
        """
        Initialize the question asker
        
        Args:
            kb_file (str): Path to knowledge base file
        """
        self.kb_file = kb_file
        self.validator = HallucinationValidator(kb_file)
        self.model_handler = LocalModelHandler()
        self.results = []
        
    def load_questions(self) -> Dict:
        """
        Load questions from knowledge base
        
        Returns:
            Dict: Questions and edge cases
        """
        try:
            with open(self.kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logging.info(f"Loaded {len(data.get('questions', []))} factual questions")
            logging.info(f"Loaded {len(data.get('edge_cases', []))} edge cases")
            
            return data
        except Exception as e:
            logging.error(f"Error loading questions: {e}")
            return {"questions": [], "edge_cases": []}
    
    def ask_question_to_model(self, question: str, retry_count: int = 0) -> str:
        """
        Ask a question to the model
        
        Args:
            question (str): Question to ask
            retry_count (int): Number of retries attempted
            
        Returns:
            str: Model's response
        """
        try:
            logging.info(f"Asking question (attempt {retry_count + 1}): {question}")
            
            # Add retry context if this is a retry
            if retry_count > 0:
                question = f"RETRY: {question}"
            
            response = self.model_handler.ask_question(question)
            
            logging.info(f"Model response: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Error asking question: {e}")
            return f"Error: {e}"
    
    def process_question(self, question: str, question_type: str = "factual") -> Dict:
        """
        Process a single question through the pipeline
        
        Args:
            question (str): Question to process
            question_type (str): Type of question (factual/edge_case)
            
        Returns:
            Dict: Processing result
        """
        result = {
            "question": question,
            "question_type": question_type,
            "timestamp": time.time(),
            "attempts": []
        }
        
        # First attempt
        model_answer = self.ask_question_to_model(question)
        validation = self.validator.validate_answer(question, model_answer)
        
        attempt = {
            "attempt_number": 1,
            "model_answer": model_answer,
            "validation": validation
        }
        result["attempts"].append(attempt)
        
        # Retry if needed
        if validation.get("retry_needed", False):
            logging.info(f"Retry needed for: {question}")
            
            # Wait a bit before retry
            time.sleep(1)
            
            # Second attempt
            retry_answer = self.ask_question_to_model(question, retry_count=1)
            retry_validation = self.validator.validate_answer(question, retry_answer)
            
            retry_attempt = {
                "attempt_number": 2,
                "model_answer": retry_answer,
                "validation": retry_validation
            }
            result["attempts"].append(retry_attempt)
        
        return result
    
    def run_hallucination_detection(self) -> List[Dict]:
        """
        Run the complete hallucination detection process
        
        Returns:
            List[Dict]: All results
        """
        logging.info("🚀 Starting Hallucination Detection Process")
        
        # Load model
        if not self.model_handler.load_model():
            logging.error("Failed to load model")
            return []
        
        # Load questions
        data = self.load_questions()
        
        all_results = []
        
        # Process factual questions
        logging.info("📚 Processing factual questions...")
        for kb_question in data.get('questions', []):
            result = self.process_question(kb_question['question'], "factual")
            result["kb_info"] = kb_question
            all_results.append(result)
            
            # Small delay between questions
            time.sleep(0.5)
        
        # Process edge cases
        logging.info("🔍 Processing edge cases...")
        for edge_case in data.get('edge_cases', []):
            result = self.process_question(edge_case['question'], "edge_case")
            result["edge_case_info"] = edge_case
            all_results.append(result)
            
            # Small delay between questions
            time.sleep(0.5)
        
        self.results = all_results
        logging.info(f"✅ Completed processing {len(all_results)} questions")
        
        return all_results
    
    def generate_summary(self) -> Dict:
        """
        Generate summary of all results
        
        Returns:
            Dict: Summary statistics
        """
        if not self.results:
            return {"error": "No results to summarize"}
        
        # Extract validation results from final attempts
        final_validations = []
        for result in self.results:
            final_attempt = result["attempts"][-1]
            final_validations.append(final_attempt["validation"])
        
        # Get summary from validator
        summary = self.validator.get_validation_summary(final_validations)
        
        # Add additional metrics
        total_attempts = sum(len(r["attempts"]) for r in self.results)
        retries_needed = sum(1 for r in self.results if len(r["attempts"]) > 1)
        
        summary.update({
            "total_attempts": total_attempts,
            "retries_needed": retries_needed,
            "retry_rate": (retries_needed / len(self.results) * 100) if self.results else 0
        })
        
        return summary
    
    def save_results(self, output_file: str = "results.json"):
        """
        Save results to JSON file
        
        Args:
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "results": self.results,
                    "summary": self.generate_summary(),
                    "timestamp": time.time()
                }, f, indent=2, ensure_ascii=False)
            
            logging.info(f"✅ Results saved to {output_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def main():
    """Main execution function"""
    print("🧠 Hallucination Detection System")
    print("=" * 50)
    
    # Initialize the system
    asker = ModelQuestionAsker()
    
    # Run the detection process
    results = asker.run_hallucination_detection()
    
    if results:
        # Generate and display summary
        summary = asker.generate_summary()
        
        print("\n📊 RESULTS SUMMARY")
        print("=" * 30)
        print(f"Total Questions: {summary['total_questions']}")
        print(f"Correct Answers: {summary['correct_answers']}")
        print(f"Hallucinations Detected: {summary['hallucinations_detected']}")
        print(f"Edge Cases: {summary['edge_cases']}")
        print(f"Out of Domain: {summary['out_of_domain']}")
        print(f"Accuracy: {summary['accuracy']:.1f}%")
        print(f"Hallucination Rate: {summary['hallucination_rate']:.1f}%")
        print(f"Retry Rate: {summary['retry_rate']:.1f}%")
        
        # Save results
        asker.save_results()
        
        print("\n✅ Process completed! Check run.log for detailed logs.")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main() 