import streamlit as st
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview  

st.set_page_config(
    page_title="MGI - Protótipo",
    layout="wide",  # Alterado para 'wide' para melhor responsividade
    initial_sidebar_state="expanded"
)

# Function for loading data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)    

if __name__ == "__main__":
    # Loading data
    df = load_data('data/SUS_Simulador_Aposentadoria.csv')

    # Navigation
    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para", ["Visão Geral", "Análises Gerais", "Modelagem de Tópicos"])

    if selection == "Visão Geral":
        # Renderizar tela de Visão Geral
        overview.render_overview()

    # Import and render the selected option
    elif selection == "Análises Gerais":
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
