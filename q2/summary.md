# Hallucination Detection System

## 🎯 Project Overview

This system implements a comprehensive hallucination detection framework for language models by comparing model responses against a curated knowledge base of factual information.

## 📁 Project Structure

```
q2/
├── kb.json              # Knowledge base with 10 factual Q-A pairs + 5 edge cases
├── model_setup.py       # Local model initialization and interaction
├── validator.py         # Hallucination detection logic
├── ask_model.py         # Main orchestration script
├── requirements.txt     # Python dependencies
├── run.log             # Execution logs (generated)
├── results.json        # Results output (generated)
└── summary.md          # This documentation
```

## 🔧 System Components

### 1. Knowledge Base (kb.json)
- **10 Factual Questions**: Covering geography, chemistry, history, biology, mathematics, economics, physics, literature
- **5 Edge Cases**: Out-of-domain, fictional, time-dependent, philosophical, and opinion-based questions
- **Structured Format**: JSON with metadata for easy validation and extension

### 2. Model Setup (model_setup.py)
- **Local Model Handler**: Uses Hugging Face Transformers for local model inference
- **Model Options**: Supports various pre-trained models (DialoGPT, GPT-2, etc.)
- **Error Handling**: Robust error management for model loading and inference

### 3. Validator (validator.py)
- **Hallucination Detection**: Compares model answers against KB using similarity metrics
- **Multiple Validation Types**:
  - `CORRECT`: Answer matches KB
  - `HALLUCINATION`: Answer differs from KB
  - `EDGE_CASE`: Question is an edge case
  - `OUT_OF_DOMAIN`: Question not in KB
- **Similarity Scoring**: Uses SequenceMatcher for fuzzy matching
- **Retry Logic**: Identifies when retries are needed

### 4. Question Asker (ask_model.py)
- **Orchestration**: Coordinates the entire detection process
- **Logging**: Comprehensive logging to run.log
- **Retry Mechanism**: Automatically retries failed questions
- **Results Management**: Saves detailed results to results.json

## 🚀 How It Works

### The Process Flow:

1. **Load Knowledge Base**: Read factual Q-A pairs and edge cases
2. **Initialize Model**: Load local language model
3. **Ask Questions**: Send each question to the model
4. **Validate Responses**: Compare model answers with KB
5. **Detect Hallucinations**: Identify mismatches and edge cases
6. **Retry if Needed**: Re-ask questions that need retry
7. **Generate Summary**: Calculate accuracy and hallucination rates
8. **Save Results**: Store all data for analysis

### Validation Logic:

```python
# For each question:
if question_in_kb:
    if similarity(model_answer, kb_answer) > 0.7:
        return "CORRECT"
    else:
        return "HALLUCINATION"
elif question_is_edge_case:
    return "EDGE_CASE"
else:
    return "OUT_OF_DOMAIN"
```

## 📊 Key Metrics

- **Accuracy**: Percentage of correct answers
- **Hallucination Rate**: Percentage of detected hallucinations
- **Retry Rate**: Percentage of questions requiring retry
- **Category Breakdown**: Performance by question category

## 🎯 Use Cases

### 1. Model Evaluation
- Assess model accuracy on factual questions
- Identify areas where model tends to hallucinate
- Compare different model versions

### 2. Quality Assurance
- Validate model responses in production
- Implement guardrails for critical applications
- Monitor model performance over time

### 3. Research & Development
- Study hallucination patterns
- Develop better detection methods
- Train models to reduce hallucinations

## 🔍 Example Results

```
📊 RESULTS SUMMARY
==============================
Total Questions: 15
Correct Answers: 12
Hallucinations Detected: 2
Edge Cases: 3
Out of Domain: 0
Accuracy: 80.0%
Hallucination Rate: 13.3%
Retry Rate: 20.0%
```

## 🛠️ Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python ask_model.py
   ```

3. **Check Results**:
   - View console output for summary
   - Check `run.log` for detailed logs
   - Review `results.json` for complete data

## 🔧 Customization

### Adding New Questions:
Edit `kb.json` to add more factual questions or edge cases.

### Changing Models:
Modify `model_setup.py` to use different Hugging Face models.

### Adjusting Validation:
Tune similarity thresholds in `validator.py` for stricter/looser validation.

## 🎓 Learning Outcomes

This project demonstrates:

1. **Hallucination Detection**: How to identify when models generate incorrect information
2. **Knowledge Base Integration**: Using structured data for validation
3. **Local Model Usage**: Working with Hugging Face Transformers
4. **System Design**: Building modular, extensible AI systems
5. **Quality Metrics**: Measuring and tracking model performance

## 🔮 Future Enhancements

- **Semantic Matching**: Use embeddings for better answer comparison
- **Confidence Scoring**: Add model confidence estimation
- **Multi-Model Comparison**: Test multiple models simultaneously
- **Real-time Monitoring**: Continuous hallucination detection
- **Advanced Edge Cases**: More sophisticated out-of-domain detection

---

*This system provides a foundation for understanding and detecting hallucinations in language models, essential for building reliable AI applications.* 