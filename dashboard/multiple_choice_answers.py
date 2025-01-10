import streamlit as st
import plotly.express as px

# Define the custom color palette
color_mapping = {
    "Concordo totalmente": '#1a9850',  # Green
    "Concordo parcialmente": '#98df8a',  # Light green
    "Não concordo, nem discordo": '#fee08b',  # Yellow
    "Discordo parcialmente": '#fc8d59',  # Orange
    "Discordo totalmente": '#d73027'  # Red
}

def create_pie_chart(responses_counts):
    
    # Create de chart
    fig = px.pie(
        names=responses_counts.index,
        values=responses_counts.values
    )

    # Assign the correct colors to each label and adjust hover
    fig.update_traces(
        marker=dict(colors=[color_mapping[response] for response in responses_counts.index]),
        hovertemplate='<br>Total de Participantes: %{value}<extra></extra>'
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig)

def split_columns_by_type(df):
    
    questions_str = []
    questions_numerical = []
    
    for column in list(df.columns)[2:-4]:  # Select only the multiple-choice columns
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
    
def render(df, topic_modeling=False, labels=[]):
    
    # Filter the questions (without numerical values in the column name)
    questions_str, questions_numerical = split_columns_by_type(df)

    # Selector for choosing the question
    selected_question = st.selectbox("Escolha uma afirmação:", questions_str)
    selected_question_index = questions_str.index(selected_question)

    # For the case where the selected main tab is topic modeling
    if topic_modeling:
        word = st.selectbox(
            "Escolha 'Considerar todos os comentários' ou selecione uma palavra do tópico como filtro:", 
            ['Considerar todos os comentários'] + labels,
            key='aba3'
        )
        if word != 'Considerar todos os comentários':
            df = df[df['clean_text'].str.contains(word, case=False, na=False)]

    # Information about the chosen question
    print_information(number_of_users=len(df), 
                      mean=df[questions_numerical[selected_question_index]].mean(),
                      std=df[questions_numerical[selected_question_index]].std()
                     )
    
    # Create the interactive pie chart
    create_pie_chart(responses_counts=df[selected_question].value_counts())