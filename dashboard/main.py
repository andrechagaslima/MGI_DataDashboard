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
        # Sidebar for topic selection
        topic_number = st.sidebar.selectbox("Selecione um Tópico:", range(1, topic_amount+1)) - 1

        # Render content based on the active topic
        topic_modeling.render(topic_number=str(topic_number),topic_amount = topic_amount)