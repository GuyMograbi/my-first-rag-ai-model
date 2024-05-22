from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat_with_bot():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("Chatbot:", bot_response)

if __name__ == "__main__":
    chat_with_bot()