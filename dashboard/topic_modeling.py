import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import multiple_choice_answers
import SUS
import overview as ov

colors_labels = {
    'criticism': "#ffcccc",
    'positive feedback': "#c8e6c9",
    'suggestion': "#F0E68C",
    'not pertinent': "#D3D3D3"
}

import plotly.graph_objects as go

def plot_comment_distribution(results_df):
    """
    Gera um gráfico de barras horizontal para exibir a distribuição de tipos de comentários.

    Parâmetros:
    results_df (pd.DataFrame): DataFrame contendo a coluna 'results' com a classificação dos comentários.
    """
    
    # Definir os tipos de comentários com rótulos em português e cores correspondentes
    categories = {
        "criticism": "Crítica",
        "positive feedback": "Elogio",
        "suggestion": "Sugestão",
        "not pertinent": "Não Pertinente"
    }
    
    colors = ["#ffcccc", "#c8e6c9", "#F0E68C", "#D3D3D3"]

    # Contagem de cada categoria
    comment_counts = {key: (results_df['results'] == key).sum() for key in categories.keys()}
    
    # Criar listas para o gráfico
    labels = list(categories.values())  # Traduzindo os nomes das categorias
    values = list(comment_counts.values())

    # Calcular porcentagens
    total_comments = sum(values)
    percentages = [(count / total_comments) * 100 if total_comments > 0 else 0 for count in values]

    # Criar gráfico de barras horizontal
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,  # Usar os rótulos em português
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in percentages],  # Adiciona porcentagem
        textposition="outside"
    ))

    # Configuração do layout
    fig.update_layout(
        title="Distribuição dos Tipos de Comentários",
        xaxis_title="Quantidade de Comentários",
        yaxis_title="Categoria",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    # Exibir no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def docs_by_word(labels, df):
    
    with open('./sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json', "r", encoding="utf-8") as file:
        data = json.load(file)

    y_pred_text = data.get("y_pred_text", [])

    comments_df = pd.read_csv('data/results_labels/flair.csv')

    # Filtrar apenas os comentários do tópico ativo usando os IDs da `df` (tópico atual)
    comments_df = comments_df[comments_df['ID'].isin(df['ID'])]

    all_comments = comments_df["Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria."]
    ids = comments_df['ID']
    sus = comments_df['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.']

    results_df = pd.DataFrame({
        "ID": ids,
        "results": y_pred_text[:len(ids)],  # Ajustar para garantir que os tamanhos correspondam
        "comments": all_comments,
        "sus": sus
    })

    results_df.to_csv('./data/results.csv', index=False, encoding="utf-8")

    # Seleção de comentários filtrados por palavra do tópico
    word = st.selectbox(
        "Escolha 'Ver todos os comentários' ou selecione uma palavra do tópico para filtrar os comentários:", 
        ['Ver todos os comentários'] + labels,
        key='aba2'
    )

    if word != 'Ver todos os comentários':
        results_df = results_df[results_df['comments'].str.contains(word, case=False, na=False)]  # Filtro pelos comentários que contêm a palavra

    # Filtrar pelo tipo de comentário
    type_of_comment = st.selectbox(
        "Escolha o tipo de comentário que deseja visualizar:", 
        ['Todos Comentários', 'Apenas Elogios', 'Apenas Críticas', 'Apenas Não Pertinente', 'Apenas Sugestões'],
        key='aba5'
    )

    word_on_topic = len(results_df) 
    if type_of_comment == 'Todos Comentários':
        plot_comment_distribution(results_df) 
    elif type_of_comment == 'Apenas Elogios':
        results_df = results_df[results_df['results'] == 'positive feedback']
    elif type_of_comment == 'Apenas Críticas':
        results_df = results_df[results_df['results'] == 'criticism']
    elif type_of_comment == 'Apenas Não Pertinente':
        results_df = results_df[results_df['results'] == 'not pertinent']
    elif type_of_comment == 'Apenas Sugestões':
        results_df = results_df[results_df['results'] == 'suggestion']

    # Exibir total de participantes filtrados
    st.markdown(
        f"<div style='text-align: right;'><strong>Total de Participantes:</strong> {len(results_df)} ({0 if word_on_topic == 0 else (len(results_df)/word_on_topic) * 100:.2f}%)</div>",
        unsafe_allow_html=True
    )

    # Exibição dos comentários filtrados
    for _, d in results_df.iterrows():
        text = d['comments']
        label = d['results']
        sus = d['sus']
        ID = d['ID']

        st.markdown(f"<div style='background-color: {colors_labels[label]}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<strong>Participante {ID} (SUS: {sus})</strong><br>{text}</div>",
                    unsafe_allow_html=True)
    
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
        path_topic_modeling=f'topic_modeling/data_num_topics/{topic_amount}/documents_scores.csv',
        data_sentiment_path='data/results_labels/flair.csv'
        )
    
    # Getting data from the topic in question
    td_sorted = df_topic_modeling[df_topic_modeling['dominant_topic'] == int(topic_number)].sort_values(by='document_score', ascending=False)
    ids = np.array(td_sorted['document_id'].tolist())
    df_data = df_data[df_data['ID'].isin(ids)]

    # Getting the words and their importance from the topic
    word_and_importance = topics_model[topic_number]
    sorted_word_and_importance = sorted(word_and_importance, key=lambda x: x[1])
    labels, values = zip(*sorted_word_and_importance)
    labels = list(labels)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([f"Análise do Tópico", "Sumarização do Tópico", "Análise dos Comentários", "Afirmações de Concordância/Discordância", "Métrica SUS por Tópico"])
    with tab1:
        get_topic_graphic(topic_number, labels, values) 
    with tab2:
         get_topic_summary(topic_number)
    with tab3:  
        docs_by_word(labels[::-1], df_data)
    with tab4:
        multiple_choice_answers.render(df_data.drop(columns=['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']), topic_modeling=True, labels=labels[::-1])      
    with tab5:
        st.title(f'Análise da Métrica SUS para os Participantes do Tópico {int(topic_number)+1}')
        SUS.render(df_data, topic_modeling=True, labels=labels[::-1])     
