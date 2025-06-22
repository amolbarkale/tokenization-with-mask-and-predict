"""
Simple Mask & Predict
Author: AI Assistant
Description: Replace 2 words with [MASK] and let AI predict them
"""

from transformers import pipeline

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
        "Machine learning [MASK] are becoming more [MASK] every year.",
        "I like to [MASK] books and [MASK] music."
    ]
    
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
            
            # Step 5: Show results
            # Check if we have multiple masks or single mask
            if isinstance(predictions[0], list):
                # Multiple masks - we get predictions for each [MASK]
                for mask_num, mask_predictions in enumerate(predictions, 1):
                    print(f"🎯 Predictions for MASK #{mask_num}:")
                    for rank, pred in enumerate(mask_predictions, 1):
                        word = pred['token_str']
                        confidence = pred['score'] * 100  # Convert to percentage
                        filled_sentence = pred['sequence']
                        
                        print(f"  {rank}. '{word}' ({confidence:.1f}% confident)")
                        print(f"     → {filled_sentence}")
                    print()
            else:
                # Single mask
                print("🎯 Predictions:")
                for rank, pred in enumerate(predictions, 1):
                    print('rank, pred:', rank, pred)
                    word = pred['token_str']
                    confidence = pred['score'] * 100
                    filled_sentence = pred['sequence']
                    
                    print(f"  {rank}. '{word}' ({confidence:.1f}% confident)")
                    print(f"     → {filled_sentence}")
        
        except Exception as e:
            print(f"❌ Error processing sentence: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("🎉 Demo completed!")
    print("💡 The AI looked at the context and predicted the most likely words!")
    print(f"{'='*50}")

# Step 6: Add a simple interactive mode
def interactive_mode():
    """Let user try their own sentences"""
    print(f"\n{'='*50}")
    print("🎮 INTERACTIVE MODE")
    print("Enter your own sentence with [MASK] tokens!")
    print("Example: I love to [MASK] and [MASK] every day.")
    print("Type 'quit' to exit")
    print(f"{'='*50}")
    
    # Load model once
    try:
        mask_filler = pipeline("fill-mask", model="bert-base-uncased")
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return
    
    while True:
        user_input = input("\n📝 Your sentence: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if '[MASK]' not in user_input:
            print("⚠️  Please include at least one [MASK] in your sentence!")
            continue
        
        try:
            print("\n🤖 AI Predictions:")
            predictions = mask_filler(user_input, top_k=3)
            
            if isinstance(predictions[0], list):
                for mask_num, mask_preds in enumerate(predictions, 1):
                    print(f"\n  MASK #{mask_num}:")
                    for rank, pred in enumerate(mask_preds, 1):
                        word = pred['token_str']
                        confidence = pred['score'] * 100
                        print(f"    {rank}. '{word}' ({confidence:.1f}%)")
            else:
                for rank, pred in enumerate(predictions, 1):
                    word = pred['token_str']
                    confidence = pred['score'] * 100
                    filled = pred['sequence']
                    print(f"  {rank}. '{word}' ({confidence:.1f}%) → {filled}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Ask if user wants to try interactive mode
    try_interactive = input("\n🎮 Want to try your own sentences? (y/n): ").lower()
    if try_interactive in ['y', 'yes']:
        interactive_mode()