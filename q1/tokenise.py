"""
Modular subword tokeniser for BPE, WordPiece, and SentencePiece (Unigram).
"""

import argparse
from typing import Dict, List
from transformers import AutoTokenizer

class Tokeniser:
    def __init__(self, model_name: str, model: str):
        self.model_name = model_name
        self.tokeniser = AutoTokenizer.from_pretrained(model)

    def encode(self, text: str) -> Dict[str, List]:
        # Avoid adding special tokens so counts match exactly.
        encoding = self.tokeniser(text, add_special_tokens=False)
        ids = encoding["input_ids"]
        tokens = self.tokeniser.convert_ids_to_tokens(ids)
        return {"tokens": tokens, "ids": ids, "count": len(tokens)}

def main():
    parser = argparse.ArgumentParser(description="Subword tokenisation demo")
    parser.add_argument(
        "--text", "-t",
        default="The cat sat on the mat because it was tired.",
        help="Sentence to tokenise"
    )
    args = parser.parse_args()

    configs = {
        "BPE (GPT-2)": "gpt2",
        "WordPiece (BERT-base-uncased)": "bert-base-uncased",
        "SentencePiece Unigram (ALBERT-base-v2)": "albert-base-v2",
    }

    results = {}
    for label, model in configs.items():
        tok = Tokeniser(label, model)
        results[label] = tok.encode(args.text)

    for label, out in results.items():
        print(f"\n=== {label} ===")
        print(f"Tokens ({out['count']}): {out['tokens']}")
        print(f"IDs: {out['ids']}")

if __name__ == "__main__":
    main()
