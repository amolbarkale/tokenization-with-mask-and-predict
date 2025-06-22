# Tokenisation & Fill-in-the-Blank

## Project Brief

This project demonstrates subword tokenisation using three popular algorithms (BPE, WordPiece, and SentencePiece Unigram) and a fill-in-the-blank prediction task using a pre-trained language model. You will:

- Tokenise a given sentence and report tokens, token IDs, and counts.
- Compare the tokenisation behavior across algorithms.
- Mask tokens in the sentence and predict them using a 7B parameter open-source model.

The code is modular and scalable for future experiments with different sentences or models.

## File Structure

```
q1/
├── tokenize.py # Script for tokenisation
├── predict.py # Script for masking and predictions
├── predictions.json # Output file for model’s top-3 fill-mask predictions
├── compare.md # Write-up comparing tokenisers and results
└── README.md # This file: project overview and instructions
```

## Prerequisites

```
- Python 3.8 or higher
- Internet connection to download pre-trained models
```

## Setup & Installation

1. **Clone the repository** (if hosted remotely):

   ```bash
   git clone <repo_url>
   cd q1
   ```

Create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\\.venv\\Scripts\\activate  # Windows
```

Install required packages:

```pip install --upgrade pip
pip install tokenizers transformers sentencepiece
```

Running the Tokenisation Script
To tokenise the default sentence:

```
python tokenize.py
```

To tokenise a custom sentence, use the --text (or -t) flag:
```
python tokenize.py --text "Your custom sentence here."
```

The script will print token lists, token IDs, and token counts for each algorithm.
