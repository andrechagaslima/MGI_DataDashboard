import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from multiple_choice_answers import print_information

categories_information = {
    'Pior Imaginável': {'min_value': 0, 'max_value': 25, 'color': '#800000'}, # Dark red
    'Ruim': {'min_value': 25, 'max_value': 50, 'color': '#d73027'}, # Red
    'Ok': {'min_value': 50, 'max_value': 70, 'color': '#fc8d59'}, # Orange
    'Bom': {'min_value': 70, 'max_value': 80, 'color': '#fee08b'}, # Yellow
    'Excelente': {'min_value': 80, 'max_value': 85, 'color': '#98df8a'}, # Light green
    'Melhor Imaginável': {'min_value': 85, 'max_value': 100, 'color': '#1a9850'} # Green
    }


def create_scale_bar(): 

    fig = go.Figure()

    for category in categories_information:
        fig.add_trace(go.Scatter(
            x=[categories_information[category]["min_value"], categories_information[category]["max_value"]],
            y=[0, 0],
            mode='lines',
            line=dict(color=categories_information[category]["color"], width=20),
            name=category,
            fill='toself'
        ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            tickvals=[0, 25, 50, 70, 80, 85, 100],
            ticktext=["0", "25", "50", "70", "80", "85", "100"],
            title="Rating Scale",
            fixedrange=True  
        ),
        yaxis=dict(
            range=[-1, 1], 
            showticklabels=False,
            fixedrange=True  
        ),
        showlegend=False,  
        height=190,
        plot_bgcolor="white",
        autosize=True,
        dragmode=False
    )

    st.plotly_chart(fig)   

def create_pie_chart(categories):
    
    fig = px.pie(
        names=categories.keys(),
        values=categories.values()
    )

    fig.update_traces(
        marker=dict(colors=[categories_information[category]['color'] for category in categories]),
        hovertemplate='<br>Total de Participantes: %{value}<extra></extra>'
    )

    st.plotly_chart(fig)

def render(df, topic_modeling=False, labels=[]):
    
    # Dictionary to store the metric value for the users
    categories = {
        "Pior Imaginável": 0,
        "Ruim": 0,
        "Ok": 0,
        "Bom": 0,
        "Excelente": 0,
        "Melhor Imaginável": 0
    }

    if topic_modeling:
        word = st.selectbox(
            "Escolha 'Considerar todas as palavras' ou selecione uma palavra do tópico como filtro:", 
            ['Considerar todas as palavras'] + labels,
            key='aba4'
            )
        if word != 'Considerar todas as palavras':
            df = df[df['clean_text'].str.contains(word, case=False, na=False)]
    
    df.loc[:, 'Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.'] = df['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.'].str.replace(',', '.', regex=False).astype(float)

    categories = {}
    for value in df['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.']:
        if value < 25:
            if "Pior Imaginável" not in categories:
                categories["Pior Imaginável"] = 0
            categories["Pior Imaginável"]+=1
        elif value < 50:
            if "Ruim" not in categories:
                categories["Ruim"] = 0
            categories["Ruim"]+=1
        elif value < 70:
            if "Ok" not in categories:
                categories["Ok"] = 0
            categories["Ok"]+=1
        elif value < 80:
            if "Bom" not in categories:
                categories["Bom"] = 0
            categories["Bom"]+=1
        elif value < 85:
            if "Excelente" not in categories:
                categories["Excelente"] = 0
            categories["Excelente"]+=1
        else: 
            if "Melhor Imaginável" not in categories:
                categories["Melhor Imaginável"] = 0
            categories["Melhor Imaginável"]+=1

    print_information(number_of_users=len(df), 
                      mean=df['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.'].mean(),
                      std=df['Some a pontuação total dos novos valores (X+Y) e multiplique por 2,5.'].std()
                     )
    
    create_scale_bar()

    create_pie_chart(categories)