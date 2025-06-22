"""
Hallucination Detection Validator
Compares model responses against knowledge base to detect hallucinations
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

class HallucinationValidator:
    def __init__(self, kb_file: str = "kb.json"):
        """
        Initialize the validator with knowledge base
        
        Args:
            kb_file (str): Path to knowledge base JSON file
        """
        self.kb_file = kb_file
        self.kb_data = self.load_knowledge_base()
        
    def load_knowledge_base(self) -> Dict:
        """Load knowledge base from JSON file"""
        try:
            with open(self.kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Knowledge base loaded: {len(data.get('questions', []))} factual questions")
            return data
        except FileNotFoundError:
            print(f"❌ Knowledge base file {self.kb_file} not found")
            return {"questions": [], "edge_cases": []}
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in {self.kb_file}")
            return {"questions": [], "edge_cases": []}
    
    def find_question_in_kb(self, question: str) -> Optional[Dict]:
        """
        Find a question in the knowledge base
        
        Args:
            question (str): Question to search for
            
        Returns:
            Optional[Dict]: KB entry if found, None otherwise
        """
        # Normalize question for comparison
        normalized_question = self.normalize_text(question)
        
        for kb_question in self.kb_data.get('questions', []):
            kb_normalized = self.normalize_text(kb_question['question'])
            
            # Check for exact match or high similarity
            if (normalized_question == kb_normalized or 
                self.calculate_similarity(normalized_question, kb_normalized) > 0.8):
                return kb_question
        
        return None
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except for important symbols
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0-1)
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def validate_answer(self, question: str, model_answer: str) -> Dict:
        """
        Validate model answer against knowledge base
        
        Args:
            question (str): Question asked
            model_answer (str): Model's response
            
        Returns:
            Dict: Validation result with details
        """
        # Find question in KB
        kb_entry = self.find_question_in_kb(question)
        
        if kb_entry:
            # Question exists in KB - check answer
            kb_answer = kb_entry['answer']
            similarity = self.calculate_similarity(
                self.normalize_text(model_answer),
                self.normalize_text(kb_answer)
            )
            
            # Determine if answer matches
            if similarity > 0.7:  # Threshold for considering answers similar
                return {
                    "status": "CORRECT",
                    "question_id": kb_entry['id'],
                    "kb_answer": kb_answer,
                    "model_answer": model_answer,
                    "similarity": similarity,
                    "category": kb_entry['category'],
                    "retry_needed": False
                }
            else:
                return {
                    "status": "HALLUCINATION",
                    "question_id": kb_entry['id'],
                    "kb_answer": kb_answer,
                    "model_answer": model_answer,
                    "similarity": similarity,
                    "category": kb_entry['category'],
                    "retry_needed": True,
                    "reason": "Answer differs from knowledge base"
                }
        else:
            # Question not in KB - check if it's an edge case
            edge_case = self.find_edge_case(question)
            
            if edge_case:
                return {
                    "status": "EDGE_CASE",
                    "edge_case_id": edge_case['id'],
                    "category": edge_case['category'],
                    "reason": edge_case['reason'],
                    "model_answer": model_answer,
                    "retry_needed": False
                }
            else:
                return {
                    "status": "OUT_OF_DOMAIN",
                    "model_answer": model_answer,
                    "retry_needed": True,
                    "reason": "Question not in knowledge base"
                }
    
    def find_edge_case(self, question: str) -> Optional[Dict]:
        """
        Check if question is an edge case
        
        Args:
            question (str): Question to check
            
        Returns:
            Optional[Dict]: Edge case entry if found
        """
        normalized_question = self.normalize_text(question)
        
        for edge_case in self.kb_data.get('edge_cases', []):
            edge_normalized = self.normalize_text(edge_case['question'])
            
            if (normalized_question == edge_normalized or 
                self.calculate_similarity(normalized_question, edge_normalized) > 0.8):
                return edge_case
        
        return None
    
    def get_validation_summary(self, results: List[Dict]) -> Dict:
        """
        Generate summary of validation results
        
        Args:
            results (List[Dict]): List of validation results
            
        Returns:
            Dict: Summary statistics
        """
        total = len(results)
        correct = sum(1 for r in results if r['status'] == 'CORRECT')
        hallucinations = sum(1 for r in results if r['status'] == 'HALLUCINATION')
        edge_cases = sum(1 for r in results if r['status'] == 'EDGE_CASE')
        out_of_domain = sum(1 for r in results if r['status'] == 'OUT_OF_DOMAIN')
        
        return {
            "total_questions": total,
            "correct_answers": correct,
            "hallucinations_detected": hallucinations,
            "edge_cases": edge_cases,
            "out_of_domain": out_of_domain,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "hallucination_rate": (hallucinations / total * 100) if total > 0 else 0
        }

def test_validator():
    """Test the validator with sample data"""
    print("🧪 Testing Hallucination Validator")
    print("=" * 50)
    
    validator = HallucinationValidator()
    
    # Test cases
    test_cases = [
        {
            "question": "What is the capital of Japan?",
            "model_answer": "Tokyo",
            "expected": "CORRECT"
        },
        {
            "question": "What is the capital of Japan?",
            "model_answer": "Beijing",
            "expected": "HALLUCINATION"
        },
        {
            "question": "What is the current population of Mars?",
            "model_answer": "There are no humans living on Mars",
            "expected": "EDGE_CASE"
        },
        {
            "question": "What is the weather like today?",
            "model_answer": "I cannot provide real-time weather information",
            "expected": "OUT_OF_DOMAIN"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Question: {test_case['question']}")
        print(f"Model Answer: {test_case['model_answer']}")
        
        result = validator.validate_answer(test_case['question'], test_case['model_answer'])
        
        print(f"Status: {result['status']}")
        print(f"Expected: {test_case['expected']}")
        print(f"✅ Pass" if result['status'] == test_case['expected'] else "❌ Fail")

if __name__ == "__main__":
    test_validator() 