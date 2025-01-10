import streamlit as st
import plotly.express as px

colors_labels = {
    -1: "#d73027",  # Negativo
    1: "#1a9850",   # Positivo
}

def render_overview():
    st.title("Visão Geral")

    # Layout com duas colunas
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Resumo da Pergunta Mais Polêmica")
        most_polemic_question = "Qual a sua opinião sobre a implementação do Simulador de Aposentadoria?"
        st.markdown(
            f"<div style='background-color: #000; padding: 15px; border-radius: 10px;'>"
            f"{most_polemic_question}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Reclamação Mais Pertinente")
        pertinent_complaint = "O simulador apresenta resultados inconsistentes em algumas simulações complexas."
        st.markdown(
            f"<div style='background-color: #B22222; padding: 15px; border-radius: 10px;'>"
            f"{pertinent_complaint}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Distribuição de Respostas Positivas e Negativas")
        sentiment_counts = {"Negativo": 40, "Positivo": 60}
        labels, values = list(sentiment_counts.keys()), list(sentiment_counts.values())
        fig = px.pie(
            values=values,
            names=labels,
            color_discrete_map={"Negativo": colors_labels[-1], "Positivo": colors_labels[1]},
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
