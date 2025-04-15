import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Upload CSV", layout="centered")
st.title("Upload CSV")

language_choice = st.selectbox("Select language (for future use):", ["Portuguese", "English [Default]"])

CSV_UPLOAD_FOLDER = "data"
TXT_UPLOAD_FOLDER = "txt_data"
os.makedirs(CSV_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TXT_UPLOAD_FOLDER, exist_ok=True)

if "upload_error" not in st.session_state:
    st.session_state.upload_error = False

if st.session_state.upload_error:
    st.session_state.upload_error = False

st.subheader("Upload your CSV file")
uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv_file")

if uploaded_csv is not None:
    try:
        header = pd.read_csv(uploaded_csv, nrows=0).columns.tolist()
        if len(header) != 12:
            st.error("The file must contain exactly 12 columns.")
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
                st.error("The first 11 columns must contain only numeric values.")
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
                    st.success("CSV accepted!")
                    save_path = os.path.join(CSV_UPLOAD_FOLDER, "dataFrame.csv")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_csv.getbuffer())
                    st.info(f"File saved in: `{save_path}`")
                else:
                    st.error("The last column must contain only string values.")
    except Exception as e:
        st.session_state.upload_error = True
        st.error(f"Error processing the file: {e}")

st.subheader("Upload your Text File")
uploaded_txt = st.file_uploader("Upload TXT", type=["txt"], key="txt_file")

if uploaded_txt is not None:
    try:
        text_content = uploaded_txt.read().decode("utf-8")
        st.text_area("Content of the text file:", text_content, height=200)
        txt_save_path = os.path.join(TXT_UPLOAD_FOLDER, uploaded_txt.name)
        with open(txt_save_path, "wb") as f:
            f.write(uploaded_txt.getbuffer())
        st.info(f"Text file saved in: `{txt_save_path}`")
    except Exception as e:
        st.error(f"Error processing the text file: {e}")
