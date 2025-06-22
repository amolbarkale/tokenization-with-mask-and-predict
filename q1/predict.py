"""
Simple Mask & Predict
Author: AI Assistant
Description: Replace 2 words with [MASK] and let AI predict them
"""

from transformers import pipeline
import json

# Simple plausibility labels based on score thresholds
def label_plausibility(score: float) -> str:
    if score >= 0.5:
        return "Very plausible"
    elif score >= 0.2:
        return "Somewhat plausible"
    else:
        return "Less plausible"


def main():
    print("🤖 Simple Mask & Predict Demo")
    print("=" * 40)
    
    print("📥 Loading AI model... (this may take a moment)")
    try:
        # Using BERT - it's reliable and well-tested Mask-Fill model
        # You can also try other models
        mask_filler = pipeline("fill-mask", model="bert-base-uncased")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running: pip install transformers torch")
        return
    
    # Step 3: Create sentences with [MASK] tokens
    # IMPORTANT: Use [MASK] not <mask> for BERT!
    sentences = [
        "The [MASK] was very [MASK] during the meeting.",
        "Python is a [MASK] programming language for [MASK] development.",
        "The weather today is [MASK] and I feel [MASK].",
        # "Machine learning [MASK] are becoming more [MASK] every year.",
        # "I like to [MASK] books and [MASK] music."
    ]
    results = []  # collect all results here
    
    # Step 4: Process each sentence
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{'='*50}")
        print(f"EXAMPLE {i}")
        print(f"{'='*50}")
        print(f"📝 Sentence: {sentence}")
        print()
        
        try:
            # Get predictions (AI fills the [MASK] tokens)
            predictions = mask_filler(sentence, top_k=3)
            example_info = {"sentence": sentence, "predictions": []}
            
            # Step 5: Show results
            # Check if we have multiple masks or single mask
            if isinstance(predictions[0], list):
                # Multiple masks - we get predictions for each [MASK]
                for mask_num, mask_predictions in enumerate(predictions, 1):
                    mask_info = {"mask_index": mask_num, "candidates": []}
                    print(f"🎯 Predictions for MASK #{mask_num}:")
                    for rank, pred in enumerate(mask_predictions, 1):
                        word = pred['token_str']
                        confidence = pred['score']
                        plaus = label_plausibility(confidence)
                        filled_sentence = pred['sequence']
                        
                        print(f"  {rank}. '{word}' ({confidence*100:.1f}% confident) -> {plaus}")
                        print(f"     → {filled_sentence}")
                        mask_info["candidates"].append({
                            "rank": rank,
                            "token": word,
                            "confidence": round(confidence,4),
                            "plausibility": plaus,
                            "sentence": filled_sentence
                        })
                    print()
                    example_info["predictions"].append(mask_info)
            else:
                # Single mask
                mask_info = {"mask_index": 1, "candidates": []}
                print("🎯 Predictions:")
                for rank, pred in enumerate(predictions, 1):
                    word = pred['token_str']
                    confidence = pred['score']
                    plaus = label_plausibility(confidence)
                    filled_sentence = pred['sequence']
                    print(f"  {rank}. '{word}' ({confidence*100:.1f}% confident) -> {plaus}")
                    print(f"     → {filled_sentence}")
                    mask_info["candidates"].append({
                        "rank": rank,
                        "token": word,
                        "confidence": round(confidence,4),
                        "plausibility": plaus,
                        "sentence": filled_sentence
                    })
                example_info["predictions"].append(mask_info)
            
            results.append(example_info)
        except Exception as e:
            print(f"❌ Error processing sentence: {e}")
            continue
    
    # Save all results to JSON
    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=2)
    print("🎉 Demo completed! Results saved to predictions.json")

if __name__ == "__main__":
    main()
