import streamlit as st
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview  

st.set_page_config(
    page_title="MGI - Protótipo",
    layout="wide",  
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)    

if __name__ == "__main__":
   
    df = load_data('data/SUS_Simulador_Aposentadoria.csv')

    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para", ["Visão Geral", "Análises", "Modelagem de Tópicos"])

    topic_amount = st.sidebar.selectbox("Selecione a Quantidade de Tópicos", (5, 10, 15))

    if selection == "Visão Geral":
        overview.render_overview(df, topic_amount)

    elif selection == "Análises":
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Afirmações de Concordância/Discordância", "Métrica SUS"])
        
        with tab1:
            multiple_choice_answers.render(df)
        with tab2:
            SUS.render(df)

    elif selection == "Modelagem de Tópicos":     
        # Sidebar for topic selection
        topic_number = st.sidebar.selectbox("Selecione um Tópico:", range(1, 11)) - 1

        # Render content based on the active topic
        topic_modeling.render(topic_number=str(topic_number))