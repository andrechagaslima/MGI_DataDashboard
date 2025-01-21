import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import multiple_choice_answers
import SUS

colors_labels = {
    -1: "#ffcccc",  #Negative
    1: "#c8e6c9"  #Positive
}

def docs_by_word(labels, df, topic_number):
    
    word = st.selectbox(
        "Escolha 'Ver todos os comentários' ou selecione uma palavra do tópico para filtrar os comentários:", 
        ['Ver todos os comentários'] + labels,
        key='aba2'
    )
    if word != 'Ver todos os comentários':
        docs_text = df[df['clean_text'].str.contains(word, case=False, na=False)][['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.', 'flair_result', 'Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.', 'ID']]
    else:
        docs_text = df[['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.', 'flair_result', 'Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.', 'ID']]

    max_words = len(docs_text)
 
    # Selecting only positive comments, only negative comments, or all comments
    type_of_comment = st.selectbox(
        "Escolha o tipo de comentário que deseja visualizar:", 
        ['Positivos e Negativos', 'Apenas Positivos', 'Apenas Negativos'],
        key='aba5'
    )
    if type_of_comment == 'Apenas Positivos':
        docs_text = docs_text[docs_text['flair_result'] == 1]
    elif type_of_comment == 'Apenas Negativos':
        docs_text = docs_text[docs_text['flair_result'] == -1]
    
    # Display chart only for "Positivos e Negativos"
    if type_of_comment == 'Positivos e Negativos':
        positive_count = len(docs_text[docs_text['flair_result'] == 1])
        negative_count = len(docs_text[docs_text['flair_result'] == -1])
     
        fig = go.Figure()
        fig.add_trace(go.Bar(
          x=[negative_count, positive_count],
          y=['Negativos', 'Positivos'],
          marker_color=['#ffcccc', '#c8e6c9'],
          orientation='h',  
          text=[negative_count, positive_count],  
          textposition='inside',  
          insidetextanchor='middle'  
        ))

        fig.update_layout(
          title={
               'text': f"Quantidade de comentários Positivos e Negativos do Tópico {int(topic_number)+1}",
               'x': 0.5,  # Centering the title
               'xanchor': 'center',
               'yanchor': 'top'
          },
          xaxis_title="Quantidade",
          plot_bgcolor="white",
          dragmode=False,  
          showlegend=False,  
          xaxis=dict(fixedrange=True, range=[0, max_words]),  
          yaxis=dict(fixedrange=True)   
        )

        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    percentage = len(docs_text)/max_words*100

    st.markdown(
        f"<div style='text-align: right;'><strong>Total de Participantes:</strong> {len(docs_text)} ({percentage:.2f}%)</div>",
        unsafe_allow_html=True
    )
    for _, d in docs_text.iterrows():
        text = d['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']
        label = d['flair_result']
        sus = d['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.']
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
            
def render(topic_number):

    topics_model, df_topic_modeling, df_data = load_data(
        path_topic_model='topic_modeling/data_topic_modeling/topics_kmeans2.json',
        path_topic_modeling='topic_modeling/data_topic_modeling/documents_scores.csv',
        data_sentiment_path='data/results_labels/flair.csv'
        )

    # Removing unnecessary data
    df_topic_modeling = df_topic_modeling.drop(columns=['Unnamed: 0'])
    df_data = df_data.dropna(subset=['clean_text']).reset_index(drop=True)

    # Getting data from the topic in question
    td_sorted = df_topic_modeling[df_topic_modeling['dominant_topic'] == int(topic_number)].sort_values(by='document_score', ascending=False)
    ids = np.array(td_sorted['document_id'].tolist())
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
        docs_by_word(labels[::-1], df_data, topic_number)
    with tab4:
        multiple_choice_answers.render(df_data.drop(columns=['Agradeço a sua participação e abro o espaço para que você possa contribuir com alguma crítica, sugestão ou elogio sobre o Simulador de Aposentadoria.']), topic_modeling=True, labels=labels[::-1])           