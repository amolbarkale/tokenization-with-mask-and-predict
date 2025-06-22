# Tokenisation & Fill-in-the-Blank

## Project Brief

This project demonstrates subword tokenisation using three popular algorithms (BPE, WordPiece, and SentencePiece Unigram) and a fill-in-the-blank prediction task using a pre-trained masked-language model. You will:

- Tokenise a given sentence and report tokens, token IDs, and counts.
- Compare the tokenisation behavior across algorithms.
- Mask tokens in sample sentences and predict them using BERT’s fill-mask pipeline, with plausibility labels.

The code is modular and scalable for future experiments with different sentences or models.

## File Structure

```bash
q1/
├── tokenize.py         # Script for subword tokenisation
├── predict.py          # Script for masking and masked-LM predictions
├── predictions.json    # Output file: detailed JSON of predictions
├── compare.md          # Analysis comparing tokenisers’ outputs
└── README.md           # This file: overview and instructions
```

## Prerequisites

- Python 3.12 or higher
- Internet connection to download pre-trained models

## Setup & Installation

1. **Clone the repository** (if hosted remotely):

   ```bash
   git clone <repo_url>
   cd q1
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\activate  # Windows
   ```

3. **Install required packages**:

   ```bash
   pip install --upgrade pip
   pip install tokenizers transformers sentencepiece torch
   ```

## Running the Tokenisation Script

To tokenise the default sentence:

```bash
python tokenize.py
```

To tokenise a custom sentence, use the `--text` (or `-t`) flag:

```bash
python tokenize.py --text "Your custom sentence here."
```

The script will print token lists, token IDs, and token counts for each algorithm.

## Running the Prediction Script

The `predict.py` script masks two tokens in a set of sample sentences and uses BERT’s `fill-mask` pipeline to predict them, assigning a plausibility label to each guess. To run:

```bash
python predict.py
```

- **Output**: Console prints top‒3 predictions per mask with confidence and plausibility.
- **JSON**: A detailed `predictions.json` is saved in the project root, containing for each example:
  - Original sentence
  - Mask index
  - Candidate token, confidence score, plausibility label, and full filled sentence

### Sample `predictions.json` entry

```json
[
  {
    "sentence": "The [MASK] was very [MASK] during the meeting.",
    "predictions": [
      {
        "mask_index": 1,
        "candidates": [
          {
            "rank": 1,
            "token": "weather",
            "confidence": 0.56,
            "plausibility": "Very plausible",
            "sentence": "The weather was very [MASK] during the meeting."
          }
        ]
      }
    ]
  }
]
```

## Next Steps

1. Populate `compare.md` with the actual token counts and observations.
2. Experiment with different tokenisers by adding entries to the `configs` dict in `tokenize.py`.
3. Tweak the sample sentences or mask positions in `predict.py` to test new contexts.
4. Extend the project: visualize results or integrate into a larger application.

---