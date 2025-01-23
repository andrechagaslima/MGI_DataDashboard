import os
import json
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def str2bool(x):
    if x.lower() in ['y', 'yes', 's', 'sim', '1', 'abacaxi']:
        return True
    return False

def check_if_out_file_exists(args):
    if os.path.exists(args.outfilename):
        print("Out dir already exist!")
        exit()

def check_if_split_exists(args):

    # if args.sel == "":
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_idxinfold.csv"
    # else:
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_"+args.sel+"_idxinfold.csv"

    saida = args.filename+".json"

    if os.path.exists(saida):
        print("Already exists selection output")
        exit()

def read_dataset(inputdir):
    df = pd.read_csv(inputdir)
    df = df.rename(columns={'Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.': 'Comentários'})
    df = df[['ID', 'Comentários']]
    df.dropna(subset=['Comentários'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def save_file(save_dir, info):
    with open(save_dir, 'w') as arquivo_json:
        json.dump(info, arquivo_json, indent=4)

def print_in_file(msg, filename):
    with open(filename, 'a') as arq:
        arq.write(msg+"\n")


def get_examples(df, prompt_dir, number_of_examples):
    with open(prompt_dir, 'r') as f:
            data = json.load(f)
    examples = data["examples"]

    labels_text_lengths  = {}
    for index, label in examples.items():
        text = df.iloc[int(index)]['Comentários']
        if label not in labels_text_lengths : 
            labels_text_lengths[label] = {index: len(text)}
        else:
            labels_text_lengths[label][index] = len(text) 
    
    texts_for_few_shot = {}
    for label in labels_text_lengths :
        labels_text_lengths [label] = dict(sorted(labels_text_lengths [label].items(), key=lambda item: item[1], reverse=True))
        for index in dict(list(labels_text_lengths[label].items())[:number_of_examples]):
            text = df.iloc[int(index)]['Comentários']
            texts_for_few_shot[int(index)] = {'text': text, 'label': label}
    
    return texts_for_few_shot 