from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import pandas as pd
import torch
import os
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
    df = pd.read_csv('../data/SUS_Simulador_Aposentadoria_pre_processado.csv')
    
    td = pd.read_csv(f'../topic_modeling/data_num_topics/{total_topics}/Resumo_Topicos_Dominantes.csv')
    papers_with_topic = td[td['dominant_topic'] == topic]['papers'].tolist()
    
    comments = df[df['ID'].isin(papers_with_topic)]
    comments = comments.reset_index(drop=True)
    comments = list(comments['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.'])
    
    print(len(comments))

    return comments

def get_standard_prompt(comments):
    """prompt = [{'role': 'system', 'content': 'You will receive a set of comments in Portuguese about a government-developed application. These comments were provided by users who were encouraged to share suggestions, critiques, or positive feedback.'},
              {'role': 'user', 'content': f'Based on the user comments below about a government-developed application, generate a well-structured and cohesive summary in a single paragraph. The summary should synthesize the most relevant points, covering suggestions, critiques, and positive feedback in a balanced and neutral manner. Avoid listing individual comments—integrate the information naturally into a flowing text. Ensure the summary remains clear, concise, and free of first-person phrases.\nThe comments are: {comments}'}]"""
    
    prompt = [
        {'role': 'system', 'content': 'You will receive a set of comments in Portuguese about a government-developed application. These comments were provided by users who were encouraged to share suggestions, critiques, or positive feedback. Your task is to summarize the information exactly as provided, without inventing any new details or adding any extra information.'},
        {'role': 'user', 'content': f'Based on the user comments below about a government-developed application, generate a well-structured and cohesive summary in a single paragraph. The summary should synthesize the most relevant points, covering suggestions, critiques, and positive feedback in a balanced and neutral manner. Do not create new information or alter the provided content. Avoid listing individual comments—integrate the information naturally into a flowing text. Ensure the summary remains clear, concise, and free of first-person phrases.\nThe comments are: {comments}'}
    ]
    
    return prompt
    
def get_additional_prompt(partial_summary, removed_comments):
    """prompt = [
    {'role': 'system', 'content': 'You will receive a partial summary of user comments regarding a government-developed application. This summary has already integrated a portion of the feedback, but additional comments will be provided for you to further enrich the summary with new information.'},
    {'role': 'user', 'content': f'Based on the partial summary provided below and the new comments about the government-developed application, update and complete the summary in a single paragraph. The new comments will add further suggestions, critiques, and positive feedback that should be integrated seamlessly. The final summary should be well-structured, cohesive, and neutral, avoiding any first-person phrases. The summary should synthesize all relevant points, maintaining clarity and conciseness.\n\nPartial summary: {partial_summary}\n\nNew comments: {removed_comments}'}]"""
    
    prompt = [
    {'role': 'system', 'content': 'You will receive a partial summary of user comments regarding a government-developed application. This summary has already integrated a portion of the feedback, but additional comments will be provided for you to further enrich the summary. Your task is to update the summary with the new information without inventing any new details or adding extra content.'},
    {'role': 'user', 'content': f'Based on the partial summary provided below and the new comments about the government-developed application, update and complete the summary in a single paragraph. The new comments will add further suggestions, critiques, and positive feedback that should be integrated seamlessly into the existing summary. Do not create new information or modify the existing content. The final summary should be well-structured, cohesive, and neutral, avoiding any first-person phrases. The summary should synthesize all relevant points, maintaining clarity and conciseness.\n\nPartial summary: {partial_summary}\n\nNew comments: {removed_comments}'}]

    return prompt

def remove_tokens_for_classification(comments, total_number_of_tokens, target_number_of_tokens=10000, batch_size=10):
        tokens_removed = 0
        current_number_of_tokens = total_number_of_tokens
        removed_comments = []
                
        while total_number_of_tokens - tokens_removed > target_number_of_tokens:    
            if not comments:
                break

            batch_size = min(batch_size, len(comments))
            removed_batch = [comments.pop() for _ in range(batch_size)]
            removed_comments.extend(removed_batch)
            
            prompt = get_standard_prompt(comments=comments)

            inputs = tokenizer.apply_chat_template(prompt,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(device)


            current_number_of_tokens = inputs['input_ids'][0].numel()
            tokens_removed = total_number_of_tokens - current_number_of_tokens
        
        return inputs.to("cuda"), removed_comments

def get_summary(comments):
    removed_comments = []
    
    prompt = get_standard_prompt(comments=comments)

    inputs = tokenizer.apply_chat_template(prompt,
                                           add_generation_prompt=True,
                                           return_dict=True,
                                           return_tensors="pt").to(device)
        
    while inputs['input_ids'].numel() > 10000:
        print('Retirada de comentários', inputs['input_ids'].numel(), end=' ')
        inputs, removed = remove_tokens_for_classification(comments=comments, 
                                                            total_number_of_tokens=inputs['input_ids'].numel()) 
        removed_comments = removed
        print(inputs['input_ids'].numel())

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
    
    while removed_comments:  
        print(summary)

        prompt = get_additional_prompt(summary, removed_comments)
         
        inputs = tokenizer.apply_chat_template(prompt,
                                               add_generation_prompt=True,
                                               return_dict=True,
                                               return_tensors="pt").to(device)
        
        if inputs['input_ids'].numel() > 10000:
            print('Retirada adicional de comentários', inputs['input_ids'].numel(), end=' ')
            inputs, removed = remove_tokens_for_classification(comments=removed_comments, 
                                                                total_number_of_tokens=inputs['input_ids'].numel())
            removed_comments = removed
            print(inputs['input_ids'].numel())
        else:
            removed_comments = []  # Se estiver dentro do limite, pode continuar
        
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
    if not os.path.exists(f'outLLM/detailed_summarization/{total_number_of_topics}'): os.makedirs(f'outLLM/detailed_summarization/{total_number_of_topics}')
    with open(f'outLLM/detailed_summarization/{total_number_of_topics}/summary_topic_{topic}.txt', 'w') as file:
        file.write(text)
        
# MAIN ---------------------------------

for total_t in [5, 10, 15]:
#for total_t in [5]:
    for t in range(total_t):
    #for t in [4, 0]:
        print(f'topic {t}')
        comments = load_data(t, total_t)
        summary = get_summary(comments)
        save_summary(summary, total_t, t)