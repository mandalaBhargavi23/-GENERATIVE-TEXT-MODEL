from transformers import GPT2LMHeadModel, GPT2Tokenizer  # it will load the model
import torch # it is used for running the model using pytorcch

# "gpt2" is the smallest version; alternatives include "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Load GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)    # Load GPT-2 model
model.eval()  # Set the model to evaluation mode 

# it will define a function for text generation
def generate_text(prompt, max_length=100):
    # Tokenize the input prompt and convert it into PyTorch tensors
    inputs = tokenizer.encode(prompt, return_tensors='pt')  # 'pt' stands for PyTorch tensor format
    
    # Generate text without calculating gradients
    with torch.no_grad():
        outputs = model.generate(
            inputs,                  # Input token IDs
            max_length=max_length,  # Maximum number of tokens in generated text
            num_return_sequences=1, # Generate only one sequence
            no_repeat_ngram_size=2, # Prevent repetition of 2-gram sequences
            do_sample=True,         # Enable sampling instead of greedy decoding for more randomness
            top_k=50,               # Keep only top 50 tokens with highest probability (Top-K sampling)
            top_p=0.95,             # Use nucleus sampling (Top-P), sampling from top 95% cumulative probability
            temperature=0.9,        # Controls randomness; lower is less random, higher is more random
        )
    
    # Decode the generated token IDs back into human-readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Remove special tokens like ``

    return generated_text  # Return the final generated string

# Example usage of the text generation function
prompt = "    "  # Initial input prompt
result = generate_text(prompt, max_length=100)  # Generate text based on prompt
print(result)  # Output the generated text

