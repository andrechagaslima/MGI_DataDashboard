import streamlit as st
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview  
import json

def remove_final_period(text):
    return text[:-1] if text.endswith('.') else text

st.set_page_config(
    page_title="MGI - Protótipo",
    layout="wide",  
    initial_sidebar_state="expanded"
)

def get_topic_title(topic_amount, topic_number):
    file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt"
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            title = file.read().strip()
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

    df_flair = pd.read_csv('data/results_labels/flair.csv')

    selected_columns = [
     "ID",
     "Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.",
     "clean_text",
     "Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria."
    ]

    df_results = df_flair[selected_columns].copy() 

    df_results.rename(columns={
     "ID": "ID",
     "Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.": "sus",
     "clean_text": "clean_comments",
     "Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.": "comments"
    }, inplace=True) 

    with open('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json', "r") as file:
     classification_data = json.load(file)

    y_pred_text = classification_data.get("y_pred_text", [])

    df_results["results"] = y_pred_text[:len(df_results)]

    df_results.to_csv('data/results.csv', index=False) 

    df = load_data('data/SUS_Simulador_Aposentadoria.csv')

    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para", ["Visão Geral", "Análises SUS", "Modelagem de Tópicos"])

    topic_amount = st.sidebar.selectbox("Selecione a Quantidade de Tópicos", (5, 10, 15))

    if selection == "Visão Geral":
        overview.render_overview(df, topic_amount)

    elif selection == "Análises SUS":
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Afirmações de Concordância/Discordância", "SUS Global"])
        
        with tab1:
            multiple_choice_answers.render(df)
        with tab2:
            SUS.render(df)

    elif selection == "Modelagem de Tópicos":     
        
        topic_titles = load_all_topic_titles(topic_amount)

        # Sidebar for topic selection # Exibe o selectbox com os títulos dos tópicos
        selected_topic_title = st.sidebar.selectbox("Selecione um Tópico:", options=topic_titles)
    
        # Recupera o índice do tópico selecionado para usar na renderização
        topic_number = topic_titles.index(selected_topic_title)

        # Render content based on the active topic
        topic_modeling.render(topic_number=str(topic_number),topic_amount = topic_amount)