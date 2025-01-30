import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import multiple_choice_answers
import SUS
import overview as ov

# Definição das cores para cada tipo de feedback
colors_labels = {
    'criticism': "#ffcccc",
    'positive feedback': "#c8e6c9",
    'suggestion': "#F0E68C",
    'not pertinent': "#D3D3D3"
}

def plot_comment_distribution(results_df):
    """ Plota a distribuição dos tipos de comentários """
    categories = {
        "criticism": "Crítica",
        "positive feedback": "Elogio",
        "suggestion": "Sugestão",
        "not pertinent": "Não Pertinente"
    }

    colors = list(colors_labels.values())

    # Contagem de cada categoria
    comment_counts = results_df["results"].value_counts().to_dict()
    
    # Criar listas para o gráfico
    labels = [categories.get(key, key) for key in categories.keys()]
    values = [comment_counts.get(key, 0) for key in categories.keys()]

    total_comments = sum(values)
    percentages = [(count / total_comments) * 100 if total_comments > 0 else 0 for count in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in percentages],
        textposition="outside"
    ))

    fig.update_layout(
        title="Distribuição dos Tipos de Comentários",
        xaxis_title="Quantidade de Comentários",
        yaxis_title="Categoria",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)

def docs_by_word(labels, df ,df_topic_modeling, topic_number):
    """ Exibe os comentários pertencentes ao tópico selecionado e pinta corretamente """
    
    # 1. Carregar os dados corretamente
    results_df = pd.read_csv('./data/results.csv', encoding="utf-8")
    comments_df = pd.read_csv('data/results_labels/flair.csv')

    # 2. Filtrar os IDs pertencentes ao tópico selecionado
    relevant_ids = df_topic_modeling[df_topic_modeling['dominant_topic'] == int(topic_number)]['document_id'].tolist()
    
    # 3. Filtrar os comentários que pertencem ao tópico
    filtered_comments = comments_df[comments_df['ID'].isin(relevant_ids)].copy()
    
    # 4. Garantir que os resultados de classificação correspondem aos comentários filtrados
    filtered_results = results_df[results_df["ID"].isin(relevant_ids)].copy()

    # 5. Combinar as informações filtradas corretamente
    results_df = filtered_comments.merge(filtered_results, on="ID", how="left")

    # 6. Aplicar seleção por palavra-chave
    word = st.selectbox(
        "Escolha 'Ver todos os comentários' ou selecione uma palavra do tópico para filtrar os comentários:", 
        ['Ver todos os comentários'] + labels,
        key='aba2'
    )

    if word != 'Ver todos os comentários':
        results_df = results_df[results_df['comments'].str.contains(word, case=False, na=False)]

    # 7. Filtragem por tipo de comentário
    type_of_comment = st.selectbox(
        "Escolha o tipo de comentário que deseja visualizar:", 
        ['Todos Comentários', 'Apenas Elogios', 'Apenas Críticas', 'Apenas Não Pertinente', 'Apenas Sugestões'],
        key='aba5'
    )

    total_participants = len(results_df)
    
    if type_of_comment == 'Todos Comentários':
        plot_comment_distribution(results_df)

    if type_of_comment == 'Apenas Elogios':
        results_df = results_df[results_df['results'] == 'positive feedback']
    elif type_of_comment == 'Apenas Críticas':
        results_df = results_df[results_df['results'] == 'criticism']
    elif type_of_comment == 'Apenas Não Pertinente':
        results_df = results_df[results_df['results'] == 'not pertinent']
    elif type_of_comment == 'Apenas Sugestões':
        results_df = results_df[results_df['results'] == 'suggestion']

    # 8. Exibir total de participantes filtrados
    st.markdown(
        f"<div style='text-align: right;'><strong>Total de Participantes:</strong> {len(results_df)} ({0 if total_participants == 0 else (len(results_df)/total_participants) * 100:.2f}%)</div>",
        unsafe_allow_html=True
    )

    # 9. Exibição dos comentários filtrados com cores corrigidas
    for _, d in results_df.iterrows():
        text = d['comments']
        label = d['results']
        sus = d['sus']
        ID = d['ID']

        color = colors_labels.get(label, "#FFFFFF")  # Cor padrão branca caso não encontre

        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([f"Análise do Tópico", "Sumarização do Tópico", "Análise dos Comentários", "Afirmações de Concordância/Discordância", "Métrica SUS do Tópico"])
    with tab1:
        get_topic_graphic(topic_number, labels, values) 
    with tab2:
         get_topic_summary(topic_number)
    with tab3:  
        docs_by_word(labels[::-1], df_data, df_topic_modeling, topic_number)
    with tab4:
        multiple_choice_answers.render(df_data.drop(columns=['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']), topic_modeling=True, labels=labels[::-1])      
    with tab5:
        st.title(f'Análise da Métrica SUS para os Participantes do Tópico {int(topic_number)+1}')
        SUS.render(df_data, topic_modeling=True, labels=labels[::-1])     
