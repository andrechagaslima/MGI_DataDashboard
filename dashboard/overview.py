import streamlit as st
import plotly.express as px
import pandas as pd

colors_labels = {
    -1: "#FFA6B1",  # Negativo
    1: "#86E886",   # Positivo
}

df_topic_modeling = pd.read_csv('topic_modeling/data_topic_modeling/documents_scores.csv')
df_flair = pd.read_csv('data/results_labels/flair.csv')

def load_topic_summary(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

def create_card_with_score(question, score, background_color):
    """Cria um card estilizado com a pergunta e a nota alinhada à direita."""
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px 15px; margin-bottom: 40px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>"
        f"<span style='font-size: 18px;'><strong>{question}</strong></span>"
        f"<span style='font-size: 24px;'><strong>{score:.2f}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def calculate_means(df):
    """Calcula as médias de todas as perguntas numéricas, excluindo elementos indesejados e ajustando índices ímpares."""
    # Selecionar colunas numéricas
    numeric_columns = df.select_dtypes(include=["number"]).columns

    # Remover os três elementos indesejados (ajustar conforme necessário)
    numeric_columns = numeric_columns[1:-2]  # Exclui a primeira coluna e as duas últimas

    # Calcular médias para as colunas restantes
    means = {}
    for idx, column in enumerate(numeric_columns):
        mean = df[column].mean()
        # Ajustar médias para índices ímpares
        if idx % 2 != 0:
            mean = 5 - mean
        means[column] = mean

    # Identificar a maior e menor média
    max_mean_question = max(means, key=means.get)
    min_mean_question = min(means, key=means.get)

    return means, (max_mean_question, means[max_mean_question]), (min_mean_question, means[min_mean_question])

def calculate_sentiment_totals(df_topic_modeling, df_flair):
    """
    Calcula o total de comentários positivos e negativos por tópico,
    retornando as taxas de positividade e negatividade.
    """
    # Mesclar os dataframes
    df_combined = df_topic_modeling.merge(
        df_flair, left_on='document_id', right_on='ID', how='inner'
    )

    # Agrupar por tópico e calcular contagens de positivos e negativos
    grouped = df_combined.groupby('dominant_topic')['flair_result'].value_counts().unstack(fill_value=0)

    # Calcular taxas positivas e negativas
    grouped['total'] = grouped[1] + grouped[-1]  # Total de comentários por tópico
    grouped['positive_rate'] = grouped[1] / grouped['total']  # Taxa de positividade
    grouped['negative_rate'] = grouped[-1] / grouped['total']  # Taxa de negatividade

    # Identificar o tópico com a maior taxa positiva e negativa
    most_positive_topic = grouped['positive_rate'].idxmax()
    most_negative_topic = grouped['negative_rate'].idxmax()

    return grouped, most_positive_topic, most_negative_topic

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
        color=labels, 
        color_discrete_map={"Negativo": colors_labels[-1], "Positivo": colors_labels[1]},
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        height=200,  
        width=200,   
        margin=dict(t=25, b=10, l=10, r=10)  
    )
    return fig

def render_positive_analysis(max, most_positive_topic, positives, negatives):
    # Primeira linha: pergunta com melhor nota
    best_question = max[0][1:].strip()  # Remove o primeiro caractere e espaços extras
    best_score = max[1]  # Nota formatada para 2 casas decimais
    st.markdown("##### Pergunta com a Melhor Nota")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#86E886"
    )

    # Segunda linha: tópicos positivos, resumo e gráfico
    col1, col2 = st.columns(2, gap="large")
    positive_summary = load_topic_summary(f'data/overview_data/positivesummary.txt')

    with col1:
        st.markdown("###### Tópico com Mais Comentários Positivos")
        create_card(
            content=f"Tópico {most_positive_topic+1}",
            background_color="#86E886"
        )

        st.markdown("###### Resumo dos Comentários Positivos")
        create_card(
            content=f"{positive_summary}",
            background_color="#86E886"
        )

    with col2:
        sentiment_counts = {"Negativo": negatives, "Positivo": positives}
        fig = create_pie_chart(sentiment_counts)
        st.markdown("###### Distribuição de Respostas Positivas", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_negative_analysis(min, most_negative_topic, positives, negatives):
    # Primeira linha: pergunta com pior nota
    worst_question = min[0][1:].strip()  # Remove o primeiro caractere e espaços extras
    worst_score = min[1]  # Nota formatada para 2 casas decimais
    st.markdown("##### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )

    # Segunda linha: tópicos negativos, resumo e gráfico
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###### Tópico com Mais Comentários Negativos")
        create_card(
            content=f"Tópico {most_negative_topic+1}",
            background_color="#FFA6B1"
        )

        negative_summary = load_topic_summary(f'data/overview_data/negativesummary.txt')
        st.markdown("###### Resumo dos Comentários Negativos")
        create_card(
            content=f"{negative_summary}",
            background_color="#FFA6B1"
        )

    with col2:
        sentiment_counts = {"Negativo": negatives, "Positivo": positives}
        fig = create_pie_chart(sentiment_counts)
        st.markdown("###### Distribuição de Respostas Negativas", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_overview(df):
    means, max, min = calculate_means(df)

    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, df_flair)

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
        render_positive_analysis(max, most_positive_topic, grouped.loc[most_positive_topic, 1], grouped.loc[most_positive_topic, -1])

    with tab2:
        render_negative_analysis(min, most_negative_topic, grouped.loc[most_negative_topic, 1], grouped.loc[most_negative_topic, -1])

if __name__ == "__main__":
    render_overview()
