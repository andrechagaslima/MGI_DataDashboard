import streamlit as st
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go

colors_labels = {
    -1: "#FFA6B1", 
    1: "#86E886",   
}

df_flair = pd.read_csv('data/results_labels/flair.csv')

def load_topic_summary(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()

def create_card_with_score(question, score, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 10px 15px; margin-bottom: 10px; border-radius: 5px; display: flex; justify-content: space-between; align-items: center;'>"
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
    """Calcula os totais de sentimentos (elogio, crítica, sugestão, não pertinente) por tópico."""
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
    grouped['positive_feedback'] = grouped.get('positive feedback', 0)
    grouped['criticism'] = grouped.get('criticism', 0)
    grouped['suggestion'] = grouped.get('suggestion', 0)
    grouped['not_pertinent'] = grouped.get('not pertinent', 0)

    grouped['positive_feedback_rate'] = (grouped['positive_feedback'] / grouped['total']) * 100
    grouped['criticism_rate'] = (grouped['criticism'] / grouped['total']) * 100

    # Identificar os tópicos com mais elogios e críticas
    most_positive_topic = grouped['positive_feedback_rate'].idxmax()
    most_critical_topic = grouped['criticism_rate'].idxmax()

    return grouped, most_positive_topic, most_critical_topic

def create_card(content, background_color):
    return st.markdown(
        f"<div style='background-color: {background_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px;'>"
        f"{content}"
        f"</div>",
        unsafe_allow_html=True,
    )

def create_percentage_bar_chart(positive_feedbacks, criticisms, suggestions, not_pertinent, title=""):
    """Cria um gráfico de barras horizontais com elogios, críticas, sugestões e não pertinentes."""
    total = positive_feedbacks + criticisms + suggestions + not_pertinent
    positive_rate = (positive_feedbacks / total) * 100 if total > 0 else 0
    criticism_rate = (criticisms / total) * 100 if total > 0 else 0
    suggestion_rate = (suggestions / total) * 100 if total > 0 else 0
    not_pertinent_rate = (not_pertinent / total) * 100 if total > 0 else 0

    fig = go.Figure()

    # Adicionar barra de críticas
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[criticism_rate],
        orientation='h',
        marker=dict(color="#FFA6B1"),
        name="Crítica",
        text=f"{criticism_rate:.2f}%",
        textposition='inside'
    ))

    # Adicionar barra de sugestões
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[suggestion_rate],
        orientation='h',
        marker=dict(color="#F0E68C"),
        name="Sugestão",
        text=f"{suggestion_rate:.2f}%",
        textposition='inside'
    ))

    # Adicionar barra de não pertinentes
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[not_pertinent_rate],
        orientation='h',
        marker=dict(color="#D3D3D3"),
        name="Não Pertinente",
        text=f"{not_pertinent_rate:.2f}%",
        textposition='inside'
    ))

    # Adicionar barra de elogios
    fig.add_trace(go.Bar(
        y=["Comentários"],
        x=[positive_rate],
        orientation='h',
        marker=dict(color="#86E886"),
        name="Elogio",
        text=f"{positive_rate:.2f}%",
        textposition='inside'
    ))

    top_margin = 50 if title else 10

    height = 200 if title else 150

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
        title=title,
        height=height,  
        margin=dict(l=10, r=10, t=top_margin, b=10),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02, 
        )
    )

    return fig

def render_topic_words(topic_number, topic_amount, x=0, title = ""):
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

    top_margin = 10

    if(title != ""):
        top_margin = 50

    fig.update_layout(
        title=title,
        xaxis_title="Relevância",
        yaxis_title="Palavras",
        plot_bgcolor="white",
        height=300,
        margin=dict(l=10, r=10, t=top_margin, b=10) 
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_words_chart_{topic_number}_{x}")

def render_overview_topics(topic_amount):
    # Criar layout de colunas para alinhar título e seletor na mesma linha
    col1, col2 = st.columns([3, 1])  # Ajuste as proporções conforme necessário

    with col1:
        st.markdown("### Análise Geral dos Tópicos")

    with col2:
        sorting_option = st.selectbox(
            " ",
            options=["Padrão", "Mais elogiado", "Mais criticado"],
            index=0,
            label_visibility="collapsed"
        )

    # Carregar os dados
    classification_data = load_classification_data('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json')
    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')

    # Calcular os sentimentos
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount)

    # **Aplicar a ordenação escolhida**
    if sorting_option == "Padrão":
        grouped = grouped.sort_index(ascending=True)
    elif sorting_option == "Mais elogiado":
        grouped = grouped.sort_values(by='positive_feedback_rate', ascending=False)
    elif sorting_option == "Mais criticado":
        grouped = grouped.sort_values(by='criticism_rate', ascending=False)

    # **Renderizar os tópicos conforme a nova ordenação**
    for topic_number, row in grouped.iterrows():
        st.markdown(f"#### Tópico {int(topic_number) + 1}")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            render_topic_words(topic_number, topic_amount, title="Relevância das Palavras do Tópico")

        with col2:
            summary_file = f'data/overview_data/topic_summary_{int(topic_number)}.txt'
            try:
                topic_summary = load_topic_summary(summary_file)
            except:
                topic_summary = "Nenhum resumo disponível para este tópico."

            st.markdown("<h6 style='font-weight: 900; margin-top: 10px;'>Resumo do Tópico</h6>", unsafe_allow_html=True)
            create_card(
                content=f"{topic_summary}",
                background_color="#f8f9fa"
            )

            positive_feedbacks = row.get('positive_feedback', 0)
            criticisms = row.get('criticism', 0)
            suggestions = row.get('suggestion', 0)
            not_pertinent = row.get('not_pertinent', 0)

            fig = create_percentage_bar_chart(
                positive_feedbacks, criticisms, suggestions, not_pertinent, "Percentual de Comentários por Sentimento"
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_chart_{int(topic_number)}")


def render_specific_topic(topic_number, topic_amount):
    st.markdown(f"### Análise do Tópico {topic_number + 1}")

    # Carregar os dados
    classification_data = load_classification_data('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json')
    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')

    # Calcular os sentimentos
    grouped, _, _ = calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount)

    # Filtrar apenas o tópico específico
    if topic_number in grouped.index:
        row = grouped.loc[topic_number]
        st.markdown(f"#### Tópico {topic_number + 1}")

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            render_topic_words(topic_number, topic_amount, title="Relevância das Palavras do Tópico")

        with col2:
            summary_file = f'data/overview_data/topic_summary_{int(topic_number)}.txt'
            try:
                topic_summary = load_topic_summary(summary_file)
            except:
                topic_summary = "Nenhum resumo disponível para este tópico."

            st.markdown("<h6 style='font-weight: 900; margin-top: 10px;'>Resumo do Tópico</h6>", unsafe_allow_html=True)
            create_card(
                content=f"{topic_summary}",
                background_color="#f8f9fa"
            )

            positive_feedbacks = row.get('positive_feedback', 0)
            criticisms = row.get('criticism', 0)
            suggestions = row.get('suggestion', 0)
            not_pertinent = row.get('not_pertinent', 0)

            fig = create_percentage_bar_chart(
                positive_feedbacks, criticisms, suggestions, not_pertinent, "Percentual de Comentários por Sentimento"
            )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"topic_chart_{int(topic_number)}")
    else:
        st.warning(f"O tópico {topic_number + 1} não foi encontrado.")

def render_response_percentages(df, question, y):
    color_mapping = {
        "Concordo Totalmente": '#1a9850',  
        "Concordo Parcialmente": '#98df8a',   
        "Não concordo, nem discordo": '#fee08b', 
        "Discordo Parcialmente": '#fc8d59',   
        "Discordo Totalmente": '#d73027'      
    }

    response_labels = {
        1: "Discordo Totalmente",
        2: "Discordo Parcialmente",
        3: "Não concordo, nem discordo",
        4: "Concordo Parcialmente",
        5: "Concordo Totalmente"
    }

    # Conta as respostas e calcula os percentuais
    response_counts = df[question].value_counts()
    total = response_counts.sum()

    # Substitui os índices pelas labels
    response_counts.index = response_counts.index.map(response_labels)

    st.markdown(
        """
        <style>
        .response-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin-top: 5px;
            border-radius: 5px;
        }
        .response-box ul {
            list-style-type: none;
            padding-left: 0;
        }
        .response-box li {
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        .color-square {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 8px;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    list_items = ""
    for response, count in response_counts.items():
        percentage = (count / total) * 100

        label = response.strip()  
        color = color_mapping.get(label, "#000000")  

        list_items += (
            f"<li>"
            f"<span class='color-square' style='background-color:{color};'></span>"
            f"<strong>{response}:</strong> {count} respostas ({percentage:.2f}%)"
            f"</li>"
        )

    st.markdown(f"""
    <div class="response-box">
      <h4>Distribuição das Respostas:</h4>
      <ul>
        {list_items}
      </ul>
    </div>
    """, unsafe_allow_html=True)

def render_positive_analysis(df, max, min, original_means):
    # Exibe as perguntas (melhor e pior) com seus percentuais de notas
    best_question = max[0][2:].strip()
    best_score = original_means[max[0]]
    
    worst_question = min[0][2:].strip()
    worst_score = original_means[min[0]]

    st.markdown("### Análise das Perguntas")
    
    # Melhor pergunta
    st.markdown("#### Pergunta com a Melhor Nota")
    create_card_with_score(
        question=best_question,
        score=best_score,
        background_color="#86E886"
    )
    render_response_percentages(df, max[0], "#86E886")

    st.markdown("---")

    # Pior pergunta
    st.markdown("#### Pergunta com a Pior Nota")
    create_card_with_score(
        question=worst_question,
        score=worst_score,
        background_color="#FFA6B1"
    )
    render_response_percentages(df, min[0], "#FFA6B1")

def render_negative_analysis(min, most_negative_topic, most_positive_topic, grouped, original_means, topic_amount, df):
    worst_question = min[0][2:].strip()
    worst_score = original_means[min[0]]

    # Seção: Análise do Tópico com Maior Percentual de Comentários Positivos
    st.markdown("### Análise do Tópico com Maior Percentual de Comentários Positivos")

    col1, col2 = st.columns(2, gap="large")
    positive_summary = load_topic_summary(f'data/overview_data/positivesummary.txt')

    with col1:
        st.markdown("#### Resumo do Tópico")
        create_card(
            content=f"Tópico {most_positive_topic + 1}",
            background_color="#86E886"
        )
        create_card(
            content=f"{positive_summary}",
            background_color="#86E886"
        )

    with col2:
        st.markdown("#### Percentual de Comentários por Sentimento")
        row = grouped.loc[most_positive_topic]
        fig = create_percentage_bar_chart(
            row['positive_feedback'],
            row['criticism'],
            row['suggestion'],
            row['not_pertinent']
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"positive_chart_{most_positive_topic}")

        st.markdown("#### Relevância das Palavras do Tópico")
        render_topic_words(most_positive_topic, topic_amount, 1)

    st.markdown("---")

    # Seção: Análise do Tópico com Maior Percentual de Comentários Negativos
    st.markdown("### Análise do Tópico com Maior Percentual de Comentários Negativos")

    col1, col2 = st.columns(2, gap="large")
    negative_summary = load_topic_summary(f'data/overview_data/negativesummary.txt')

    with col1:
        st.markdown("#### Resumo do Tópico")
        create_card(
            content=f"Tópico {most_negative_topic + 1}",
            background_color="#FFA6B1"
        )
        create_card(
            content=f"{negative_summary}",
            background_color="#FFA6B1"
        )

    with col2:
        st.markdown("#### Percentual de Comentários por Sentimento")
        row = grouped.loc[most_negative_topic]
        fig = create_percentage_bar_chart(
            row['positive_feedback'],
            row['criticism'],
            row['suggestion'],
            row['not_pertinent']
        )
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"negative_chart_{most_negative_topic}")

        st.markdown("#### Relevância das Palavras do Tópico")
        render_topic_words(most_negative_topic, topic_amount, 2)

def load_classification_data(file_path):
    """Carrega os dados de classificação de sentimentos a partir de um arquivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['y_pred_text']

def render_overview(df, topic_amount):
    classification_data = load_classification_data('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json') 

    means, max, min, original_means = calculate_means(df)

    df_topic_modeling = pd.read_csv(f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv')      

    grouped, most_positive_topic, most_negative_topic = calculate_sentiment_totals(df_topic_modeling, classification_data, topic_amount)

    st.markdown(
        "<h1 style='text-align: center; font-size: 28px; padding: 0px 10px 10px 10px;'>Visão Geral</h1>",
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

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Mantendo as três abas
    tab1, tab2, tab3 = st.tabs(["Análise Geral dos Tópicos", "Análise das Afirmações (Melhor/Pior)", "Análise dos Tópicos (Melhor/Pior)"])

    with tab1:
        render_overview_topics(topic_amount)

    with tab2:
        render_positive_analysis(df, max, min, original_means)

    with tab3:
        render_negative_analysis(
            min, 
            most_negative_topic, 
            most_positive_topic, 
            grouped, 
            original_means, 
            topic_amount, 
            df
        )

if __name__ == "__main__":
    render_overview()
