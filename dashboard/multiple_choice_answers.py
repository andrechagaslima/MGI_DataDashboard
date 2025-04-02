import streamlit as st
import plotly.express as px

def split_columns_by_type(df):
    questions_str = []
    questions_numerical = []

    for column in list(df.columns)[2:-4]: 
        if not any(str(value) in column for value in range(10)):
            questions_str.append(column)
        else:
            questions_numerical.append(column)

    return questions_str, questions_numerical

def print_information(number_of_users, mean, std):
    st.markdown(
        f"<div><strong>Total de Participantes:</strong> {number_of_users}</div>"
        f"<div><strong>Valor Médio:</strong> {mean:.2f}</div>"
        f"<div><strong>Desvio Padrão:</strong> {std:.2f}</div>",
        unsafe_allow_html=True
    )

color_mapping = {
    "Concordo totalmente": '#1a9850',     # Green
    "Concordo parcialmente": '#98df8a',   # Dark green
    "Não concordo, nem discordo": '#fee08b',  # Yellow
    "Discordo parcialmente": '#fc8d59',   # Orange
    "Discordo totalmente": '#d73027'      # Red
}

def create_response_info_box(responses_counts):

    total = responses_counts.sum()
    
    st.markdown(
        """
        <style>
        .gray-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin-top: 15px;
            border-radius: 5px;
        }
        .gray-box ul {
            list-style-type: none;
            padding-left: 0;
        }
        .gray-box li {
            margin-bottom: 8px;
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
    for response, count in responses_counts.items():
        percentage = (count / total) * 100
        color = color_mapping.get(response, "#000000")  
        
        list_items += (
            f"<li>"
            f"<span class='color-square' style='background-color:{color};'></span>"
            f"<strong>{response}:</strong> {count} respostas ({percentage:.2f}%)"
            f"</li>"
        )
    
    st.markdown(f"""
    <div class="gray-box">
        <h4>Distribuição das Respostas:</h4>  
        <ul>
            {list_items}
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render(df, topic_modeling=False, labels=[]):
    questions_str, questions_numerical = split_columns_by_type(df)

    selected_question = st.selectbox("Escolha uma afirmação:", questions_str)
    selected_question_index = questions_str.index(selected_question)

    if topic_modeling:
        word = st.selectbox(
            "Escolha 'Considerar todos os comentários' ou selecione uma palavra do tópico como filtro:", 
            ['Considerar todos os comentários'] + labels,
            key='aba3'
        )
        if word != 'Considerar todos os comentários':
            df = df[df['clean_text'].str.contains(word, case=False, na=False)]

    print_information(
        number_of_users=len(df), 
        mean=df[questions_numerical[selected_question_index]].mean(),
        std=df[questions_numerical[selected_question_index]].std()
    )

    response_counts = df[selected_question].value_counts()

    create_response_info_box(response_counts)
