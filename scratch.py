from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id)
# kwargs to from_pretrained: torch_dtype=torch.bfloat16, device_map="auto"

# Prepare inputs
myinput=[
    [{"role": "user", "content": "1 + 1 = "}],
    [{"role": "user", "content": "Introduce C++ in one short sentence less than 10 words."}],
    [{"role": "user", "content": "Who was the first president of the United States? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the capital of France ? Answer in less than 10 words."}],
    [{"role": "user", "content": "Why is the sky blue ? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the meaning of life? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the best way to learn a new language? Answer in less than 10 words."}],
    [{"role": "user", "content": "When is the best time to plant a tree? Answer in less than 10 words."}],
    [{"role": "user", "content": "What is the best way to cook an egg? Answer in less than 10 words."}],
    [{"role": "user", "content": "Which is the best programming language? Answer in less than 10 words."}]
]
texts = tokenizer.apply_chat_template(myinput, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(texts, padding="longest", return_tensors="pt")

# Generate outputs
gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=512, 
    pad_token_id=tokenizer.eos_token_id, 
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

# Decode
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]