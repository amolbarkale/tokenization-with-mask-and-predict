# Comparison of Tokenisation Algorithms

**Sentence:** "The cat sat on the mat because it was tired."

| Algorithm                          | Tokens                                                                                      | IDs                                                                   | Count |
| ---------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----- |
| **BPE (GPT-2)**                    | `['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']`  | `[464, 3797, 3332, 319, 262, 2603, 780, 340, 373, 10032, 13]`         | 11    |
| **WordPiece (BERT-base-uncased)**  | `['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']`           | `[1996, 4937, 2938, 2006, 1996, 13523, 2138, 2009, 2001, 5458, 1012]` | 11    |
| **SentencePiece Unigram (ALBERT)** | `['▁the', '▁cat', '▁sat', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.']` | `[14, 2008, 847, 27, 14, 4277, 185, 32, 23, 4117, 9]`                 | 11    |

---

## Brief Note on Split Differences

All three tokenisers yield 11 tokens for this simple sentence but differ in their segmentation strategies:

 **BPE** 
- Starts with single letters and glues together the pairs that appear most often.

- Marks each new word with “Ġ” to show where spaces were.

- Rare parts of words get split off if they don’t appear often.

 **WordPiece**
- Looks at the whole training text and asks: “Can I cover most words using as few pieces as possible?”

- Keeps common words whole and only splits when a word is rare or unseen.

 **SentencePiece Unigram** 
- Begins with a giant list of possible word pieces, then throws out the ones people hardly ever use.

- Uses “▁” to mark the start of each word.

- Ends up balancing between small merges (like BPE) and whole words (like WordPiece).
