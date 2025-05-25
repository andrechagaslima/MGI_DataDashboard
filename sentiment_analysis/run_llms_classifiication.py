import json
import time
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils.args import args_llm
from src.llms.llm_for_few_shot import LLM
from src.utils.geral import read_dataset, save_file, get_examples

SEED = 2024

def run_classification():
    args, info = args_llm()

    df = read_dataset(args.inputdir)
    texts_for_few_shot = get_examples(df, args.prompt_dir, args.number_of_examples)

    # Cria modelo
    llm = LLM(llm_method = args.llm_method,
              prompt_dir = args.prompt_dir)
    
    llm.set_model(texts_for_few_shot)
    
    # Classificação
    classification_start_time = time.time()
    print("Predict!")
    y_pred_text = llm.predict(df)
    classification_end_time = time.time()

    info["time_to_classify"] = classification_end_time - classification_start_time
    info["time_to_classify_avg"] = (classification_end_time - classification_start_time)/len(df)
    info["y_pred_text"] = y_pred_text
    info["prompt"] = llm.system_prompt
    
    print(json.dumps(info, indent=4))
    save_file(args.outfilename, info)

run_classification()