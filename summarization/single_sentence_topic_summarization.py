from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import pandas as pd
import torch
import os
import json
import random
import numpy

# REPRODUTIBILIDADE ---------------------
SEED=2025
random.seed(SEED); torch.manual_seed(SEED); numpy.random.seed(seed=SEED)

# MODEL ---------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token_access = "seu_token"

model = AutoModelForCausalLM.from_pretrained(name_model, 
                                             torch_dtype="auto", 
                                             device_map=device,
                                             offload_buffers=True,
                                             token=token_access,
                                             trust_remote_code=True,
                                             use_cache=False,
                                             load_in_4bit=True,
                                             use_safetensors=True)

tokenizer = AutoTokenizer.from_pretrained(name_model, 
                                          torch_dtype="auto", 
                                          device_map=device, 
                                          offload_buffers=True, 
                                          token=token_access, 
                                          use_safetensors=True,
                                          trust_remote_code=True) 

tokenizer.pad_token = tokenizer.eos_token        
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# SUMMARIZATION -------------------------

def load_data(topic, total_topics):
    with open(f'outLLM/concise_summarization/{total_topics}/summary_topic_{topic}.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    return text

def get_prompt(text):
    """prompt = [
        {'role': 'system', 'content': 'You will receive a brief summary of user comments about a government-developed application, along with key words that represent the topic of these comments. Your task is to generate a concise, informative description that captures the core feedback, focusing on the most relevant points without additional explanation.'},
        {'role': 'user', 'content': f'Generate a concise description, up to 6 words, that highlights the key feedback, such as critiques, suggestions, or positive comments. Focus on the most relevant and precise points without repeating or describing the application. Only provide the description—no additional information.\n\nSummary: {text}'}]"""

    prompt = [
    {'role': 'system', 'content': 'You will receive a brief summary of user comments about a government-developed application, along with key words that represent the topic of these comments. Your task is to generate a concise, informative description that captures the core feedback. Do not add any new details or explanations—focus only on the relevant points from the summary provided.'},
    {'role': 'user', 'content': f'Generate a concise description, up to 6 words, that highlights the key feedback, such as critiques, suggestions, or positive comments. Focus on the most relevant and precise points without repeating or describing the application. Do not create new information. Only provide the description—no additional information.\n\nSummary: {text}'}]
    
    return prompt

def get_summary(text):    
    prompt = get_prompt(text=text)

    inputs = tokenizer.apply_chat_template(prompt,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(device)

    set_seed(SEED)
    outputs = model.generate(inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             max_new_tokens=1000,
                             eos_token_id=terminators,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True,
                             temperature=0.05,
                             top_p=0.9,
                             use_cache=False)
    
    summary = outputs[0][inputs['input_ids'].shape[-1]:]
    summary = tokenizer.decode(summary, skip_special_tokens=True)

    torch.cuda.empty_cache()
                
    return summary  


def save_summary(text, total_number_of_topics, topic):
    if not os.path.exists(f'outLLM/single_sentence/{total_number_of_topics}'): os.makedirs(f'outLLM/single_sentence/{total_number_of_topics}')
    with open(f'outLLM/single_sentence/{total_number_of_topics}/summary_topic_{topic}.txt', 'w') as file:
        file.write(text)
        
# MAIN ---------------------------------

for total_t in [5, 10, 15]:
#for total_t in [5]:
    for t in range(total_t):
    #for t in [4, 0]:
        print(f'topic {t}')
        text = load_data(t, total_t)
        summary = get_summary(text)
        save_summary(summary, total_t, t)