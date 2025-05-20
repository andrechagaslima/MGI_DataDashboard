import streamlit as st
import os
import pandas as pd
import multiple_choice_answers
import SUS
import topic_modeling
import overview
import json

st.set_page_config(page_title="MGI - Protótipo", layout="wide", initial_sidebar_state="expanded")
CSV_UPLOAD_FOLDER = "data"
TXT_UPLOAD_FOLDER = "txt_data"
os.makedirs(CSV_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TXT_UPLOAD_FOLDER, exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def csv_upload_page():
    st.title("Upload CSV")
    st.selectbox("Selecione o idioma (para uso futuro):", ["Portuguese", "English [Default]"])
    st.subheader("Envie seu arquivo CSV")
    uploaded_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"], key="csv_file")
    success = False
    if uploaded_csv is not None:
        try:
            header = pd.read_csv(uploaded_csv, nrows=0).columns.tolist()
            if len(header) != 12:
                st.error("O arquivo deve conter exatamente 12 colunas.")
            else:
                dtypes = {header[11]: str}
                uploaded_csv.seek(0)
                df = pd.read_csv(uploaded_csv, dtype=dtypes)
                valid_numbers = True
                for col in df.columns[:11]:
                    try:
                        pd.to_numeric(df[col])
                    except Exception:
                        valid_numbers = False
                        break
                if not valid_numbers:
                    st.error("As primeiras 11 colunas devem conter apenas valores numéricos.")
                else:
                    last_col = df.columns[11]
                    def is_numeric_string(x):
                        x_str = str(x).strip()
                        try:
                            float(x_str)
                            return not any(c.isalpha() for c in x_str)
                        except:
                            return False
                    if df[last_col].dropna().apply(lambda x: not is_numeric_string(x)).all():
                        st.success("CSV aceito!")
                        save_path = os.path.join(CSV_UPLOAD_FOLDER, "dataFrame.csv")
                        with open(save_path, "wb") as f:
                            f.write(uploaded_csv.getbuffer())
                        st.info(f"Arquivo salvo em: `{save_path}`")
                        success = True
                    else:
                        st.error("A última coluna deve conter apenas valores do tipo string.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    st.subheader("Envie seu arquivo TXT com as stopwords")
    txt_stopwords = st.file_uploader("Selecione um arquivo TXT", type=["txt"], key="txt_stopwords")
    if txt_stopwords is not None:
        try:
            text_content = txt_stopwords.read().decode("utf-8")
            st.text_area("Conteúdo do arquivo de texto:", text_content, height=200)
            txt_save_path = os.path.join(TXT_UPLOAD_FOLDER, txt_stopwords.name)
            with open(txt_save_path, "wb") as f:
                f.write(txt_stopwords.getbuffer())
            st.info(f"Arquivo de texto salvo em: `{txt_save_path}`")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo de texto: {e}")

    st.subheader("Envie seu arquivo TXT com as frases e sua análise de sentimento")
    txt_sentiment_analysis = st.file_uploader("Selecione um arquivo TXT", type=["txt"], key="txt_sentiment_analysis")
    if txt_sentiment_analysis is not None:
        try:
            text_content = txt_sentiment_analysis.read().decode("utf-8")

            # Salvar o arquivo
            txt_save_path = os.path.join(TXT_UPLOAD_FOLDER, txt_sentiment_analysis.name)
            with open(txt_save_path, "wb") as f:
                f.write(txt_sentiment_analysis.getbuffer())

            # Extrair os sentimentos
            lines = text_content.strip().splitlines()
            labels = []
            for line in lines:
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    label = parts[1].strip().strip('"').strip(",")
                    labels.append(label)

            if labels:
                from collections import Counter
                label_counts = Counter(labels)
                min_count = min(label_counts.values())

                selected_quantity = st.selectbox(
                    "Quantidade de frases por sentimento:",
                    options=list(range(1, min_count + 1)),
                    index=min_count - 1
                )

                # Salvar no session_state
                st.session_state["selected_quantity_per_label"] = selected_quantity

        except Exception as e:
            st.error(f"Erro ao processar o arquivo de texto: {e}")
    return success

def pre_processing_df():
    df = load_data("data/dataFrame.csv")

    df['X'] = df.iloc[:, [1, 3, 5, 7, 9]].sum(axis=1) - 5
    df['Y'] = 25 - df.iloc[:, [2, 4, 6, 8, 10]].sum(axis=1) 
    df['sus'] = df.iloc[:, [12, 13]].sum(axis=1) * 2.5

    df.columns.values[11] = "comments"

    df.to_csv('data/dataFrame.csv', index=False)

def get_topic_title(topic_amount, topic_number):
    file_path = f"summarization/outLLM/single_sentence/{topic_amount}/summary_topic_{int(topic_number)}.txt"
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            title = file.read().replace('"', '').strip()
        return title[:-1] if title.endswith('.') else title
    except FileNotFoundError:
        return f"Arquivo não encontrado: {file_path}"

def load_all_topic_titles(topic_amount):
    titles = []
    for topic_number in range(topic_amount):
        title = get_topic_title(topic_amount, topic_number)
        titles.append(title)
    return titles

def main_app():
    pre_processing_df()

    #RETIRAR DEPOIS

    df_flair = load_data('data/results_labels/flair.csv')

    df = load_data('data/dataFrame.csv')

    selected_columns = [
        "ID",
        "sus",
        "comments"
    ]

    df_results = df[selected_columns].copy() 

    df_results = df_results[df_results["comments"].notna()].reset_index(drop=True)

    df_results['clean_text'] = df_flair['clean_text']

    with open('sentiment_analysis/resources/outLLM/sentiment_analysis/prompt4/3_few_shot/classification.json', "r") as file:
        classification_data = json.load(file)

    y_pred_text = classification_data.get("y_pred_text", [])

    df_results["results"] = y_pred_text[:len(df_results)]

    df_results.to_csv('data/results.csv', index=False) 

    st.sidebar.title("Navegação")
    selection = st.sidebar.radio("Ir para", ["Visão Geral", "Análises SUS", "Modelagem de Tópicos"])

    topic_amount = st.sidebar.selectbox("Selecione a Quantidade de Tópicos", (5, 10, 15))


    if selection == "Visão Geral":
        overview.render_overview(df, topic_amount)

    elif selection == "Análises SUS":
        tab1, tab2 = st.tabs(["Afirmações de Concordância/Discordância", "SUS Global"])
        
        with tab1:
            multiple_choice_answers.render(df)
        with tab2:
            SUS.render(df)

    elif selection == "Modelagem de Tópicos":     
        
        topic_titles = load_all_topic_titles(topic_amount)

        selected_topic_title = st.sidebar.selectbox("Selecione um Tópico:", options=topic_titles)
    
        topic_number = topic_titles.index(selected_topic_title)

        topic_modeling.render(topic_number=str(topic_number),topic_amount = topic_amount)

if "csv_uploaded" not in st.session_state:
    st.session_state["csv_uploaded"] = False

if not st.session_state["csv_uploaded"]:
    success = csv_upload_page()
    if success:
        st.session_state["csv_uploaded"] = True
    else:
        st.stop()

main_app()


