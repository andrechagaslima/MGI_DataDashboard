import streamlit as st
import plotly.express as px
import pandas as pd

# Define cores para os labels
colors_labels = {
    -1: "#d73027",  # Negativo
    1: "#1a9850",   # Positivo
}

df_topic_modeling = pd.read_csv('topic_modeling/data_topic_modeling/documents_scores.csv')

def create_card_with_score(question, score, background_color):
    """Cria um card estilizado com a pergunta e a nota alinhada à direita."""
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 15px; margin-bottom: 40px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>"
        f"<span style='font-size: 18px;'><strong>{question}</strong></span>"
        f"<span style='font-size: 24px;'><strong>{score}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def create_card(content, background_color):
    """Cria um card estilizado para exibição em Streamlit."""
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>"
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )

def create_pie_chart(sentiment_counts):
    """Cria um gráfico de pizza para distribuição de sentimentos."""
    labels, values = list(sentiment_counts.keys()), list(sentiment_counts.values())
    fig = px.pie(
        values=values,
        names=labels,
        color_discrete_map={"Negativo": colors_labels[-1], "Positivo": colors_labels[1]},
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=350, width=350)
    return fig

def render_positive_analysis():
    # Primeira linha: pergunta com melhor nota
    best_question = "Qual a sua opinião sobre a interface do aplicativo?"
    best_score = 4.8  # Exemplo de nota
    st.markdown("##### Pergunta com a Melhor Nota")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#98FB98"
    )

    # Segunda linha: tópicos positivos, resumo e gráfico
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###### Tópico com Mais Comentários Positivos")
        create_card(
            content="Interface do Aplicativo",
            background_color="#98FB98"
        )

        st.markdown("###### Resumo dos Comentários Positivos")
        create_card(
            content="Os usuários elogiaram a simplicidade e a clareza da interface.",
            background_color="#98FB98"
        )

    with col2:
        sentiment_counts = {"Negativo": 20, "Positivo": 80}
        fig = create_pie_chart(sentiment_counts)
        st.markdown("###### Distribuição de Respostas Positivas", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_negative_analysis():
    # Primeira linha: pergunta com pior nota
    worst_question = "Qual a sua opinião sobre o Simulador de Aposentadoria?"
    worst_score = 2.1  # Exemplo de nota
    st.markdown("##### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFC0CB"
    )

    # Segunda linha: tópicos negativos, resumo e gráfico
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###### Tópico com Mais Comentários Negativos")
        create_card(
            content="Performance do Simulador",
            background_color="#FFC0CB"
        )

        st.markdown("###### Resumo dos Comentários Negativos")
        create_card(
            content="Muitos usuários relataram lentidão ao carregar os resultados.",
            background_color="#FFB6C1"
        )

    with col2:
        sentiment_counts = {"Negativo": 60, "Positivo": 40}
        fig = create_pie_chart(sentiment_counts)
        st.markdown("###### Distribuição de Respostas Negativas", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_overview(df):
    """Renderiza a visão geral com tabs para análises positivas e negativas."""
    st.markdown(
        "<h1 style='text-align: center; font-size: 28px;'>Visão Geral</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            f"<div style='background-color: whitesmoke; padding: 5px; border-radius: 10px; text-align: center;'>"
            f"<strong>Total de Participantes:</strong> {len(df)}</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div style='background-color: whitesmoke; padding: 5px; border-radius: 10px; text-align: center;'>"
            f"<strong>Total de Comentários:</strong> {len(df_topic_modeling)}</div>",
            unsafe_allow_html=True
        )

    tab1, tab2 = st.tabs(["Análise Positiva", "Análise Negativa"])

    with tab1:
        render_positive_analysis()

    with tab2:
        render_negative_analysis()

if __name__ == "__main__":
    render_overview()
