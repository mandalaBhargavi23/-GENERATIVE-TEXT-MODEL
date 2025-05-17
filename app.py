import streamlit as st  # For creating the web app interface
from transformers import GPT2LMHeadModel, GPT2Tokenizer  
import torch                                  
# Load GPT-2 model and tokenizer only once using Streamlit's caching mechanism
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")     # Load the tokenizer for GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2")       # Load the GPT-2 model
    model.eval()                                           # Set model to evaluation mode 
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Define a function to generate text based on a prompt
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')   # Convert input text into token IDs
    with torch.no_grad():                                    # Disable gradient calculation for faster inference
        outputs = model.generate(
            inputs,
            max_length=max_length,         # Maximum length of the generated text
            num_return_sequences=1,        # Only generate one output sequence
            no_repeat_ngram_size=2,        # Avoid repeating 2-word phrases
            do_sample=True,                # Enable random sampling
            top_k=50,                      # Use top-k sampling (choose from top 50 tokens)
            top_p=0.95,                    # Use top-p (nucleus) sampling (cumulative probability = 0.95)
            temperature=0.9,               # Add randomness; lower = more predictable, higher = more creative
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # Convert token IDs back to readable text

# Streamlit UI section
st.title("🧠 GPT-2 Text Generator")  # App title
st.write("Enter a prompt and generate text using a pre-trained GPT-2 model.")  # Description

# Prompt input area
prompt = st.text_area("Enter your prompt:", "     ")  # Text box for user input

# Slider to choose maximum text length
max_len = st.slider("Max Length", min_value=50, max_value=300, value=100, step=10)

# Button to trigger text generation
if st.button("Generate"):
    with st.spinner("Generating..."):                      # Show spinner while processing
        output = generate_text(prompt, max_length=max_len)  # Call text generation function
    st.subheader("Generated Text:")                       # Output section title
    st.write(output)                                      # Display the generated text
