import streamlit as st
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go


colors_labels = {
    -1: "#FFA6B1",  # Negative
    1: "#86E886",   # Positive
}

df_topic_modeling = pd.read_csv('topic_modeling/data_topic_modeling/documents_scores.csv')
df_flair = pd.read_csv('data/results_labels/flair.csv')

def load_topic_summary(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

def create_card_with_score(question, score, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px 15px; margin-bottom: 40px; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;'>"
        f"<span style='font-size: 18px;'><strong>{question}</strong></span>"
        f"<span style='font-size: 24px;'><strong>{score:.2f}</strong></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def calculate_means(df):
    numeric_columns = df.select_dtypes(include=["number"]).columns

    numeric_columns = numeric_columns[1:-2]  

    means = {}
    for idx, column in enumerate(numeric_columns):
        mean = df[column].mean()
        if idx % 2 != 0:
            mean = 6 - mean
        means[column] = mean

    max_mean_question = max(means, key=means.get)
    min_mean_question = min(means, key=means.get)

    return means, (max_mean_question, means[max_mean_question]), (min_mean_question, means[min_mean_question])

def calculate_sentiment_totals(df_topic_modeling, df_flair):

    df_combined = df_topic_modeling.merge(
        df_flair, left_on='document_id', right_on='ID', how='inner'
    )

    grouped = df_combined.groupby('dominant_topic')['flair_result'].value_counts().unstack(fill_value=0)

    grouped['total'] = grouped[1] + grouped[-1]  
    grouped['positive_rate'] = grouped[1] / grouped['total']  
    grouped['negative_rate'] = grouped[-1] / grouped['total']  

    most_positive_topic = grouped['positive_rate'].idxmax()
    most_negative_topic = grouped['negative_rate'].idxmax()

    return grouped, most_positive_topic, most_negative_topic

def create_card(content, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>"
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )

def create_pie_chart(sentiment_counts):
    labels, values = list(sentiment_counts.keys()), list(sentiment_counts.values())
    fig = px.pie(
        values=values,
        names=labels,
        color=labels, 
        color_discrete_map={"Negativo": colors_labels[-1], "Positivo": colors_labels[1]},
    )
    fig.update_traces(textinfo="percent")
    fig.update_layout(
        height=200,  
        width=200,   
        margin=dict(t=25, b=10, l=10, r=10)  
    )
    return fig

def render_overview_topics():
    st.markdown("### Análise Geral dos Tópicos")
    
    # Carregar os dados de tópicos
    try:
        with open('topic_modeling/data_topic_modeling/topics_kmeans2.json', 'r') as file:
            topics_model = json.load(file)
    except Exception as e:
        st.error(f"Erro ao carregar os dados dos tópicos: {e}")
        return

    # Iterar sobre cada tópico
    for topic_number, words_importance in topics_model.items():
        # Verificar se os dados do tópico estão no formato correto
        if not isinstance(words_importance, list) or len(words_importance) == 0:
            st.warning(f"Tópico {topic_number} não contém dados válidos.")
            continue

        # Ordenar palavras por relevância
        valid_words_importance = [
            item for item in words_importance if isinstance(item, list) and len(item) == 2
        ]
        if len(valid_words_importance) == 0:
            st.warning(f"Tópico {topic_number} não possui palavras com relevância.")
            continue

        labels, values = zip(*valid_words_importance)
        
        # Criar gráfico de barras horizontal
        fig = go.Figure(go.Bar(
            y=list(labels),
            x=list(values),
            orientation='h',
            marker=dict(color='#3BCBDE')
        ))

        fig.update_layout(
            title=f"Tópico {int(topic_number) + 1}: Relevância das Palavras",
            xaxis_title="Relevância",
            yaxis_title="Palavras",
            plot_bgcolor="white"
        )

        # Exibir o gráfico
        st.plotly_chart(fig, use_container_width=True)


def render_positive_analysis(max, most_positive_topic, positives, negatives):
    
    best_question = max[0][2:].strip() 

    best_score = max[1]  
    st.markdown("##### Pergunta com a Melhor Nota")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#86E886"
    )

    col1, col2 = st.columns(2, gap="large")
    positive_summary = load_topic_summary(f'data/overview_data/positivesummary.txt')

    with col1:
        st.markdown("###### Tópico com Mais Comentários Positivos")
        create_card(
            content=f"Tópico {most_positive_topic+1}",
            background_color="#86E886"
        )

        st.markdown(f"###### Resumo dos Comentários do Tópico {most_positive_topic+1}")
        create_card(
            content=f"{positive_summary}",
            background_color="#86E886"
        )

    with col2:
        sentiment_counts = {"Negativo": negatives, "Positivo": positives}
        fig = create_pie_chart(sentiment_counts)
        st.markdown(f"###### Taxa de Comentários Positivos e Negativos do Tópico {most_positive_topic+1}", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_negative_analysis(min, most_negative_topic, positives, negatives):
    
    worst_question = min[0][1:].strip()  
    worst_score = min[1]  
    st.markdown("##### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("###### Tópico com Mais Comentários Negativos")
        create_card(
            content=f"Tópico {most_negative_topic+1}",
            background_color="#FFA6B1"
        )

        negative_summary = load_topic_summary(f'data/overview_data/negativesummary.txt')
        st.markdown(f"###### Resumo dos Comentários do Tópico {most_negative_topic+1}")
        create_card(
            content=f"{negative_summary}",
            background_color="#FFA6B1"
        )

    with col2:
        sentiment_counts = {"Negativo": negatives, "Positivo": positives}
        fig = create_pie_chart(sentiment_counts)
        st.markdown(f"###### Taxa de Comentários Positivos e Negativos do Tópico {most_negative_topic+1}", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, height=400)

def render_overview(df):
    means, max, min = calculate_means(df)

    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, df_flair)

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

    tab1, tab2, tab3 = st.tabs(["Análise Geral dos Tópicos","Análise Positiva", "Análise Negativa"])

    with tab1: 
        render_overview_topics()  

    with tab2:
        render_positive_analysis(max, most_positive_topic, grouped.loc[most_positive_topic, 1], grouped.loc[most_positive_topic, -1])

    with tab3:
        render_negative_analysis(min, most_negative_topic, grouped.loc[most_negative_topic, 1], grouped.loc[most_negative_topic, -1])

if __name__ == "__main__":
    render_overview()
