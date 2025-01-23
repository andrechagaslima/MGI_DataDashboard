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

def calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount):
    """Calcula os totais de sentimentos (positivo, negativo, neutro) por tópico."""
    # Garantir que o número de classificações corresponde ao número de tópicos no DataFrame
    if len(classification_data) > len(df_topic_modeling):
        classification_data = classification_data[:len(df_topic_modeling)]
    elif len(classification_data) < len(df_topic_modeling):
        raise ValueError("O número de classificações é menor que o número de tópicos no DataFrame.")

    # Adicionar as classificações ao DataFrame de tópicos
    df_topic_modeling['classification'] = classification_data

    # Agrupar por tópico dominante e contar classificações
    grouped = df_topic_modeling.groupby('dominant_topic')['classification'].value_counts().unstack(fill_value=0)

    # Calcular os totais
    grouped['total'] = grouped.sum(axis=1)
    grouped['positive'] = grouped.get('positive', 0)
    grouped['neutral'] = grouped.get('neutral', 0)
    grouped['negative'] = grouped.get('negative', 0)

    grouped['positive_rate'] = (grouped['positive'] / grouped['total']) * 100
    grouped['neutral_rate'] = (grouped['neutral'] / grouped['total']) * 100
    grouped['negative_rate'] = (grouped['negative'] / grouped['total']) * 100

    # Identificar os tópicos mais positivos, negativos e neutros
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

def create_percentage_bar_chart(positives, neutrals, negatives):
    """Cria um gráfico de barras horizontais com positivos, neutros e negativos."""
    total = positives + neutrals + negatives
    positive_rate = (positives / total) * 100 if total > 0 else 0
    neutral_rate = (neutrals / total) * 100 if total > 0 else 0
    negative_rate = (negatives / total) * 100 if total > 0 else 0

    fig = go.Figure()

    # Adicionar barra de negativos
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[negative_rate],
        orientation='h',
        marker=dict(color="#FFA6B1"),
        name="Negativo",
        text=f"{negative_rate:.2f}%",
        textposition='inside'
    ))

    # Adicionar barra de neutros
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[neutral_rate],
        orientation='h',
        marker=dict(color="#F0E68C"),  # Cor para neutros
        name="Neutro",
        text=f"{neutral_rate:.2f}%",
        textposition='inside'
    ))

    # Adicionar barra de positivos
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[positive_rate],
        orientation='h',
        marker=dict(color="#86E886"),
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
        title="Porcentagem de Comentários (Positivos, Neutros e Negativos)",
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True
    )

    return fig

def render_topic_words(topic_number, topic_amount, x=0):
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

    # Adicionar uma chave única ao gráfico
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_words_chart_{topic_number}_{x}")


def render_overview_topics(topic_amount):
    st.markdown("### Análise Geral dos Tópicos")

    # Carregar os dados
    classification_data = load_classification_data('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt2/3_few_shot/classification.json')
    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')

    # Calcular os sentimentos
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount)

    for topic_number, row in grouped.iterrows():
        st.markdown(f"#### Tópico {int(topic_number) + 1}")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown("##### Relevância das Palavras")
            render_topic_words(topic_number, topic_amount)

        with col2:
            positives = row.get('positive', 0)
            neutrals = row.get('neutral', 0)
            negatives = row.get('negative', 0)
            fig = create_percentage_bar_chart(positives, neutrals, negatives)
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_chart_{int(topic_number)}")


def render_response_percentages(df, question):
    """Exibe as respostas e percentuais com base no dataset enviado, substituindo os números pelas respostas completas."""
    # Mapeamento de números para respostas completas (com base no dataset)
    response_labels = {
        1: "Discordo Totalmente",
        2: "Discordo Parcialmente",
        3: "Não concordo, nem discordo",
        4: "Concordo Parcialmente",
        5: "Concordo Totalmente"
    }

    # Conta as respostas e calcula os percentuais
    response_counts = df[question].value_counts(normalize=True) * 100

    # Ordena conforme a ordem natural dos índices
    response_counts = response_counts.sort_index()

    responses_html = ""
    for resposta, percentual in response_counts.items():
        # Substitui o índice pelo rótulo do mapeamento
        label = response_labels.get(resposta, resposta)  # Garante que algo será exibido, mesmo sem mapeamento
        responses_html += (
            f"<div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>"
            f"<span>{label}</span>"
            f"<span>{percentual:.2f}%</span>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 40px;'>"
        f"<strong>Distribuição das Respostas:</strong>"
        f"<div style='margin-top: 10px;'>{responses_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def render_positive_analysis(max, most_positive_topic, positives, neutrals, negatives, original_means, topic_amount, df):
    best_question = max[0][2:].strip()
    best_score = original_means[max[0]]

    st.markdown("##### Pergunta com a Melhor Nota")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#86E886"
    )

    # Exibir a distribuição das respostas
    render_response_percentages(df, max[0])

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
        fig = create_percentage_bar_chart(positives, neutrals, negatives)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"positive_chart_{most_positive_topic}")

        render_topic_words(most_positive_topic, topic_amount, 1)  # Gráfico de palavras já corrigido



def render_negative_analysis(min, most_negative_topic, positives, neutrals, negatives, original_means, topic_amount, df):
    worst_question = min[0][2:].strip()
    worst_score = original_means[min[0]]

    st.markdown("##### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )

    # Exibir a distribuição das respostas
    render_response_percentages(df, min[0])

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
        fig = create_percentage_bar_chart(positives, neutrals, negatives)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"negative_chart_{most_negative_topic}")

        render_topic_words(most_negative_topic, topic_amount, 2)  # Gráfico de palavras já corrigido

def load_classification_data(file_path):
    """Carrega os dados de classificação de sentimentos a partir de um arquivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['y_pred_text']

def render_overview(df, topic_amount):
    classification_data = load_classification_data('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt2/3_few_shot/classification.json') 

    means, max, min, original_means = calculate_means(df)

    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')      

    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount)

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
        render_positive_analysis(max, most_positive_topic, grouped.loc[most_positive_topic, 'positive'], grouped.loc[most_positive_topic, 'neutral'], grouped.loc[most_positive_topic, 'negative'], original_means, topic_amount, df)

    with tab3:
        render_negative_analysis(min, most_negative_topic, grouped.loc[most_negative_topic, 'positive'], grouped.loc[most_negative_topic, 'neutral'], grouped.loc[most_negative_topic, 'negative'], original_means, topic_amount, df)

if __name__ == "__main__":
    render_overview()
