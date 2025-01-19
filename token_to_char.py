from transformers import GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode text into tokens
text = """Hello,a"`world!"""
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")

# Inspect the first few tokens
for token in tokens[:5]:
    decoded_token = tokenizer.decode([token])
    print(f"Token {token} decodes to: '{decoded_token}'")

# Inspect token 1 specifically
token_1_decoded = tokenizer.decode([1])
print(f"Token 1 decodes to: '{token_1_decoded}'")