import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import multiple_choice_answers
import SUS
import overview as ov

# Exemplo de dicionário de cores para cada tipo de sentimento
colors_labels = {
    'POSITIVE': '#c8e6c9',  # verde claro
    'NEGATIVE': '#ffcccc', # vermelho claro
    'NEUTRAL': '#F0E68C'   # amarelo claro
}

def docs_by_word(labels, df, topic_number, topic_amount):
    # Carrega dados
    classification_data = ov.load_classification_data(
        'sentiment_analysis/resources/outLLM/sentiment_analysis/prompt2/3_few_shot/classification.json'
    )
    df_topic_modeling = pd.read_csv(
        f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv'
    )

    # Agrupa os dados de sentimento por tópico
    grouped, _, _ = ov.calculate_sentiment_totals(
        df_topic_modeling, classification_data, topic_amount
    )

    positives = grouped.loc[int(topic_number), 'positive']
    neutrals = grouped.loc[int(topic_number), 'neutral']
    negatives = grouped.loc[int(topic_number), 'negative']

    # Caixa de seleção para palavra específica
    word = st.selectbox(
        "Escolha 'Ver todos os comentários' ou selecione uma palavra do tópico para filtrar os comentários:", 
        ['Ver todos os comentários'] + labels,
        key='aba2'
    )

    if word != 'Ver todos os comentários':
        # Filtra somente comentários que contenham a 'word'
        docs_text = df[
            df['clean_text'].str.contains(word, case=False, na=False)
        ][[
            'Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.',
            'flair_result',
            'Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.',
            'ID'
        ]]
    else:
        # Mostra todos os comentários
        docs_text = df[[
            'Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.',
            'flair_result',
            'Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.',
            'ID'
        ]]

    max_words = len(docs_text)

    # Caixa de seleção para filtrar o tipo de comentário
    type_of_comment = st.selectbox(
        "Escolha o tipo de comentário que deseja visualizar:", 
        [
            'Positivos, Negativos e Neutros',
            'Apenas Positivos',
            'Apenas Negativos',
            'Apenas Neutros'
        ],
        key='aba5'
    )
    
    # Exibe gráfico de barras se a opção for "Positivos, Negativos e Neutros"
    if type_of_comment == 'Positivos, Negativos e Neutros':
        positive_rate = (positives / len(docs_text) * 100) if len(docs_text) > 0 else 0
        neutral_rate  = (neutrals / len(docs_text) * 100) if len(docs_text) > 0 else 0
        negative_rate = (negatives / len(docs_text) * 100) if len(docs_text) > 0 else 0

        positive_text = f"{positives} ({positive_rate:.2f}%)"
        neutral_text  = f"{neutrals} ({neutral_rate:.2f}%)"
        negative_text = f"{negatives} ({negative_rate:.2f}%)"  

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[negatives, positives, neutrals],
            y=['Negativos', 'Positivos', 'Neutros'],
            marker_color=['#ffcccc', '#c8e6c9', '#F0E68C'],
            orientation='h',
            text=[negative_text, positive_text, neutral_text],
            textposition='inside',
            insidetextanchor='middle'
        ))

        fig.update_layout(
            title={
                'text': f"Quantidade de comentários Positivos e Negativos do Tópico {int(topic_number)+1}",
                'x': 0.5,  # centraliza o título
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Quantidade",
            plot_bgcolor="white",
            dragmode=False,
            showlegend=False,
            xaxis=dict(fixedrange=True, range=[0, len(docs_text)]),
            yaxis=dict(fixedrange=True)
        )

        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    percentage = (len(docs_text) / max_words * 100) if max_words > 0 else 0
    st.markdown(
        f"<div style='text-align: right;'><strong>Total de Participantes:</strong> {len(docs_text)} ({percentage:.2f}%)</div>",
        unsafe_allow_html=True
    )

    # Exibe cada comentário filtrado
    for _, d in docs_text.iterrows():
        text = d['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']
        label = d['flair_result']
        sus = d['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.']
        ID = d['ID']

        st.markdown(
            f"<div style='background-color: {colors_labels.get(label, '#FFFFFF')}; "
            f"padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            f"<strong>Participante {ID} (SUS: {sus})</strong><br>{text}</div>",
            unsafe_allow_html=True
        )

    ############################################################################
    #                  SEPARANDO OS COMENTÁRIOS POSITIVOS POR TÓPICO          #
    ############################################################################
    df_topic_modeling.rename(columns={'papers': 'ID'}, inplace=True) 
    # 1) Fazemos um merge para termos a coluna 'Dominant_Topic' no mesmo dataframe de docs_text
    #    (assumindo que ambos têm a coluna ID para ligação).
    #    Note que se você quiser *todos os comentários*, não só os filtrados na tela,
    #    troque "docs_text" por "df" no merge.
    df_merged = pd.merge(
        docs_text,
        df_topic_modeling[['ID', 'dominant_topic']],
        on='ID',
        how='left'
    )
    st.table(df_merged)
    # 2) Criamos um dicionário para armazenar, em cada tópico, os comentários POSITIVE
    positive_comments_by_topic = {}
    df_merged['flair_result'] = df_merged['flair_result'].astype(str)  # Converte para string
    df_merged['flair_result'] = df_merged['flair_result'].str.upper()   # Transforma em maiúsculo

    for t in range(topic_amount):
     subset = df_merged[
        (df_merged['dominant_topic'] == t) &
        (df_merged['flair_result'] == 'POSITIVE')
    ]
    positive_comments_by_topic[t] = subset

    # Se quiser retornar esse dicionário para uso futuro:
    return positive_comments_by_topic
    
def get_topic_graphic(topic_number, labels, values):

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color='#3BCBDE')
    ))

    fig.update_layout(
        title={
            'text': f'Análise das Palavras do Tópico {int(topic_number)+1} e Suas Respectivas Pontuações*',
            'font': {'size': 24},
            'x': 0.5,  # Centers the title
            'y': 0.9,
            'xanchor': 'center'
        },
        xaxis_title={
            'text': 'Pontuação',
            'font': {'color': 'black'}
        },
        yaxis_title={
            'text': 'Palavra',
            'font': {'color': 'black'}
        },
        xaxis=dict(
            tickfont=dict(color='black')  
        ),
        yaxis=dict(
            tickfont=dict(color='black')  
        ),
        dragmode=False,
        plot_bgcolor="white",  
        autosize=True
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    st.plotly_chart(fig, use_container_width=True)

    text = """
<p style="font-size: 16px; color: black">*A pontuação exibida reflete a <strong>importância</strong> de <strong>cada palavra</strong> dentro do tópico. 
Assim, quanto <strong>maior</strong> a <strong>pontuação</strong> de uma palavra, <strong>mais relevante</strong> ela é para o tópico em questão.</p>"""

    st.markdown(text, unsafe_allow_html=True)


@st.cache_data
def load_topic_summary(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read()
    
def get_topic_summary(topic_number):

    topic_summary = load_topic_summary(f'topic_summary/summary_topic_{topic_number}.txt')
    
    st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                f"\n\n{topic_summary}"
                f"</div>",
                unsafe_allow_html=True)

@st.cache_data
def load_data(path_topic_model, path_topic_modeling, data_sentiment_path):
    return (json.load(open(path_topic_model, 'r')),
            pd.read_csv(path_topic_modeling),
            pd.read_csv(data_sentiment_path)
    )
            
def render(topic_number, topic_amount):

    topics_model, df_topic_modeling, df_data = load_data(
        path_topic_model=f'topic_modeling/data_num_topics/{topic_amount}/topics_{topic_amount}.json',
        path_topic_modeling=f'topic_modeling/data_num_topics/{topic_amount}/Resumo_Topicos_Dominantes.csv',
        data_sentiment_path='data/results_labels/flair.csv'
        )
    
    # Getting data from the topic in question
    td_sorted = df_topic_modeling[df_topic_modeling['dominant_topic'] == int(topic_number)]
    ids = np.array(td_sorted['papers'].tolist())
    df_data = df_data[df_data['ID'].isin(ids)]

    # Getting the words and their importance from the topic
    word_and_importance = topics_model[topic_number]
    sorted_word_and_importance = sorted(word_and_importance, key=lambda x: x[1])
    labels, values = zip(*sorted_word_and_importance)
    labels = list(labels)

    tab1, tab2, tab3, tab4 = st.tabs([f"Análise do Tópico", "Sumarização do Tópico", "Análise dos Comentários", "Afirmações de Concordância/Discordância"])
    with tab1:
        get_topic_graphic(topic_number, labels, values)
        st.divider()
        st.title(f'Análise da Métrica SUS para os Participantes do Tópico {int(topic_number)+1}')
        SUS.render(df_data, topic_modeling=True, labels=labels[::-1])  
    with tab2:
         get_topic_summary(topic_number)
    with tab3:  
        docs_by_word(labels[::-1], df_data, topic_number, topic_amount)
    with tab4:
        multiple_choice_answers.render(df_data.drop(columns=['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']), topic_modeling=True, labels=labels[::-1])           