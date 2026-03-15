from transformers import pipeline
import os

# Loading the Pre-trained Deep Learning Model
# Instead of building an AI from scratch, I'm using Hugging Face's 'transformers' library.
print("Loading the AI Summarizer model... (This might take a minute or two to download the first time!)\n", flush=True)

# Using a standard model that is great for fast text summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    # Setting rules for the AI
    # I don't want the summary to be too long or too short, so I set max and min limits.
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Interactive testing menu for the terminal
if __name__ == "__main__":
    print("📝 Welcome to the AI Text Summarizer! 📝", flush=True)
    print("Paste your long English or French articles here to get a quick summary.", flush=True)
    
    while True:
        user_input = input("\nPaste your text here (or type 'q' to quit):\n")
        
        if user_input.lower() == 'q':
            print("Exiting system. Au revoir!\n", flush=True)
            break
            
        # Summarization only makes sense for longer texts (more than 40 words)
        if len(user_input.split()) < 40:
            print("⚠️ This text is too short! Please paste a longer paragraph to summarize.", flush=True)
            continue
            
        print("\n⚙️ AI is analyzing and summarizing the text...\n", flush=True)
        try:
            result = summarize_text(user_input)
            print("--- AI SUMMARY ---", flush=True)
            print(result, flush=True)
            print("------------------", flush=True)
        except Exception as e:
            print(f"An error occurred: {e}", flush=True)
