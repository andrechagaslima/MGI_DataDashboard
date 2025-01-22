import streamlit as st
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go

colors_labels = {
    -1: "#FFA6B1",  # Negative
    1: "#86E886",   # Positive
}

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
    original_means = {}  

    for idx, column in enumerate(numeric_columns):
        mean = df[column].mean()
        original_means[column] = mean  
        if idx % 2 != 0:
            mean = 6 - mean
        means[column] = mean

    max_mean_question = max(means, key=means.get)
    min_mean_question = min(means, key=means.get)

    return (
        means,
        (max_mean_question, means[max_mean_question]),
        (min_mean_question, means[min_mean_question]),
        original_means,  
    )

def calculate_sentiment_totals(df_topic_modeling, df_flair, topic_amount):

    df_combined = df_topic_modeling.merge(
        df_flair, left_on='papers', right_on='ID', how='inner'
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

def create_percentage_bar_chart(positives, negatives, reverse_colors=False):
    total = positives + negatives
    positive_rate = (positives / total) * 100
    negative_rate = (negatives / total) * 100

    fig = go.Figure()

    if reverse_colors:
        # Adiciona a barra de positivos (à direita)
        fig.add_trace(go.Bar(
            y=["Comentários"],
            x=[positive_rate],
            orientation='h',
            marker=dict(color=colors_labels[1]),
            name="Positivo",
            text=f"{positive_rate:.2f}%",
            textposition='inside'
        ))

        # Adiciona a barra de negativos (à esquerda)
        fig.add_trace(go.Bar(
            y=["Comentários"],
            x=[negative_rate],
            orientation='h',
            marker=dict(color=colors_labels[-1]),
            name="Negativo",
            text=f"{negative_rate:.2f}%",
            textposition='inside'
        ))
    else:
        # Adiciona a barra de negativos (à esquerda)
        fig.add_trace(go.Bar(
            y=["Comentários"],
            x=[negative_rate],
            orientation='h',
            marker=dict(color=colors_labels[-1]),
            name="Negativo",
            text=f"{negative_rate:.2f}%",
            textposition='inside'
        ))

        # Adiciona a barra de positivos (à direita)
        fig.add_trace(go.Bar(
            y=["Comentários"],
            x=[positive_rate],
            orientation='h',
            marker=dict(color=colors_labels[1]),
            name="Positivo",
            text=f"{positive_rate:.2f}%",
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            title="Porcentagem (%)",
            range=[0, 100],
            ticksuffix='%'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=14)
        ),
        plot_bgcolor="white",
        title="Porcentagem de Comentários Positivos e Negativos",
        height=150,  # Aumenta a espessura da barra
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )

    return fig

def render_topic_words(topic_number, topic_amount):
    try:
       with open(f'topic_modeling/data_num_topics/{topic_amount}/topics_{topic_amount}.json', 'r') as file:
            topics_model = json.load(file)
    except Exception as e:
        st.error(f"Erro ao carregar os dados dos tópicos: {e}")
        return

    words_importance = topics_model.get(str(topic_number), [])
    valid_words_importance = [
        item for item in words_importance if isinstance(item, list) and len(item) == 2
    ]
    if len(valid_words_importance) == 0:
        st.warning(f"Tópico {topic_number} não possui palavras com relevância.")
        return

    valid_words_importance = sorted(valid_words_importance, key=lambda x: x[1], reverse=True)
    labels, values = zip(*valid_words_importance)

    fig = go.Figure(go.Bar(
        y=list(reversed(labels)),
        x=list(reversed(values)),
        orientation='h',
        marker=dict(color='#3BCBDE')
    ))

    fig.update_layout(
        title=f"Tópico {int(topic_number) + 1}: Relevância das Palavras",
        xaxis_title="Relevância",
        yaxis_title="Palavras",
        plot_bgcolor="white",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10) 
    )

    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def render_overview_topics(topic_amount):
    st.markdown("### Análise Geral dos Tópicos")
    
    # Carregar os dados de tópicos de acordo com a quantidade selecionada
    try:
        file_path = f'topic_modeling/data_num_topics/{topic_amount}/topics_{topic_amount}.json'
        with open(file_path, 'r') as file:
            topics_model = json.load(file)
    except Exception as e:
        st.error(f"Erro ao carregar os dados dos tópicos para {topic_amount + 1} tópicos: {e}")
        return

    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')

    topic_numbers = list(topics_model.keys())
    
    # Carregar os dados de sentimentos
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, df_flair, topic_amount)
    
    # Iterar pelos tópicos
    for topic_number in topic_numbers:
        st.markdown(f"#### Tópico {int(topic_number) + 1}")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            words_importance = topics_model[topic_number]
            
            # Verificar se os dados do tópico estão no formato correto
            valid_words_importance = [
                item for item in words_importance if isinstance(item, list) and len(item) == 2
            ]
            if len(valid_words_importance) == 0:
                st.warning(f"Tópico {topic_number} não possui palavras com relevância.")
                continue
            
            # Ordenar palavras por relevância
            valid_words_importance = sorted(valid_words_importance, key=lambda x: x[1], reverse=True)
            labels, values = zip(*valid_words_importance)
            
            # Criar gráfico de relevância de palavras
            fig_words = go.Figure(go.Bar(
                y=list(reversed(labels)),  # Reverter para que a mais relevante fique no topo
                x=list(reversed(values)),
                orientation='h',
                marker=dict(color='#3BCBDE')  # Azul pastel escuro
            ))

            fig_words.update_layout(
                title="Relevância das Palavras",
                xaxis_title="Relevância",
                yaxis_title="Palavras",
                plot_bgcolor="white",
                height=400
            )

            # Exibir o gráfico de palavras
            st.plotly_chart(fig_words, use_container_width=True)
        
        with col2:
            # Adicionar gráfico de sentimentos para o tópico atual
            if int(topic_number) in grouped.index:
                positives = grouped.loc[int(topic_number), 1]
                negatives = grouped.loc[int(topic_number), -1]
                fig_sentiments = create_percentage_bar_chart(positives, negatives, reverse_colors=True)

                st.markdown(
                    """
                    <div style="padding-top: 36px; display: flex; flex-direction: column; justify-content: center; height: 100%;">
                    """,
                    unsafe_allow_html=True,
                )

                st.plotly_chart(fig_sentiments, use_container_width=True, config={"staticPlot": True})
                st.markdown("</div>", unsafe_allow_html=True)

def render_positive_analysis(max, most_positive_topic, positives, negatives, original_means, topic_amount):
    best_question = max[0][2:].strip()
    best_score = original_means[max[0]]  
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
            content=f"Tópico {most_positive_topic + 1}",
            background_color="#86E886"
        )

        st.markdown(f"###### Resumo dos Comentários do Tópico {most_positive_topic + 1}")
        create_card(
            content=f"{positive_summary}",
            background_color="#86E886"
        )

    with col2:
        # Garantindo que cada gráfico tenha uma chave única
        fig = create_percentage_bar_chart(positives, negatives, reverse_colors=True)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"positives_bar_chart_{most_positive_topic}")

        # Adicionando gráfico de palavras com uma chave única
        render_topic_words(most_positive_topic, topic_amount)

def render_negative_analysis(min, most_negative_topic, positives, negatives, original_means, topic_amount):

    worst_question = min[0][2:].strip()
    worst_score = original_means[min[0]]  
    
    st.markdown("##### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )

    col1, col2 = st.columns(2, gap="large")
    negative_summary = load_topic_summary(f'data/overview_data/negativesummary.txt')

    with col1:
        st.markdown("###### Tópico com Mais Comentários Negativos")
        create_card(
            content=f"Tópico {most_negative_topic + 1}",
            background_color="#FFA6B1"
        )

        st.markdown(f"###### Resumo dos Comentários do Tópico {most_negative_topic + 1}")
        create_card(
            content=f"{negative_summary}",
            background_color="#FFA6B1"
        )

    with col2:
        fig = create_percentage_bar_chart(positives, negatives)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

        # Adicionando gráfico de palavras
        render_topic_words(most_negative_topic, topic_amount)

def render_overview(df, topic_amount):
    means, max, min, original_means = calculate_means(df)

    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')      

    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, df_flair, topic_amount)

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

    tab1, tab2, tab3 = st.tabs(["Análise Geral dos Tópicos", "Análise Positiva", "Análise Negativa"])

    with tab1:
        render_overview_topics(topic_amount)  

    with tab2:
        render_positive_analysis(max, most_positive_topic, grouped.loc[most_positive_topic, 1], grouped.loc[most_positive_topic, -1], original_means, topic_amount)

    with tab3:
        render_negative_analysis(min, most_negative_topic, grouped.loc[most_negative_topic, 1], grouped.loc[most_negative_topic, -1], original_means, topic_amount)

if __name__ == "__main__":
    render_overview()
