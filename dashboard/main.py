import streamlit as st
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview  
import json

st.set_page_config(
    page_title="MGI - Protótipo",
    layout="wide",  
    initial_sidebar_state="expanded"
)

def pre_processing_df():
    df = load_data("data/dataFrame.csv")

    df['X'] = df.iloc[:, [1, 3, 5, 7, 9]].sum(axis=1) - 5
    df['Y'] = 25 - df.iloc[:, [2, 4, 6, 8, 10]].sum(axis=1) 
    df['sus'] = df.iloc[:, [12, 13]].sum(axis=1) * 2.5

    df.columns.values[11] = "comments"

    df.to_csv('data/dataFrame.csv', index=False)

def get_topic_title(topic_amount, topic_number):
    file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt"
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            title = file.read().replace('"', '').strip()
        return title[:-1] if title.endswith('.') else title
    except FileNotFoundError:
        return f"Arquivo não encontrado: {file_path}"

def load_all_topic_titles(topic_amount):
    titles = []
    for topic_number in range(topic_amount):
        title = get_topic_title(topic_amount, topic_number)
        titles.append(title)
    return titles

@st.cache_data
def load_data(path):
    return pd.read_csv(path)    

if __name__ == "__main__":

    pre_processing_df()

    #RETIRAR DEPOIS

    df_flair = load_data('data/results_labels/flair.csv')

    df = load_data('data/dataFrame.csv')

    selected_columns = [
        "ID",
        "sus",
        "comments"
    ]

    df_results = df[selected_columns].copy() 

    df_results = df_results[df_results["comments"].notna()].reset_index(drop=True)

    df_results['clean_text'] = df_flair['clean_text']

    with open('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json', "r") as file:
        classification_data = json.load(file)

    y_pred_text = classification_data.get("y_pred_text", [])

    df_results["results"] = y_pred_text[:len(df_results)]

    df_results.to_csv('data/results.csv', index=False) 

    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para", ["Visão Geral", "Análises SUS", "Modelagem de Tópicos"])

    topic_amount = st.sidebar.selectbox("Selecione a Quantidade de Tópicos", (5, 10, 15))


    if selection == "Visão Geral":
        overview.render_overview(df, topic_amount)

    elif selection == "Análises SUS":
        tab1, tab2 = st.tabs(["Afirmações de Concordância/Discordância", "SUS Global"])
        
        with tab1:
            multiple_choice_answers.render(df)
        with tab2:
            SUS.render(df)

    elif selection == "Modelagem de Tópicos":     
        
        topic_titles = load_all_topic_titles(topic_amount)

        selected_topic_title = st.sidebar.selectbox("Selecione um Tópico:", options=topic_titles)
    
        topic_number = topic_titles.index(selected_topic_title)

        topic_modeling.render(topic_number=str(topic_number),topic_amount = topic_amount)