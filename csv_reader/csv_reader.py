import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Upload CSV", layout="centered")
st.title("Upload CSV")

UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if "upload_error" not in st.session_state:
    st.session_state.upload_error = False

if st.session_state.upload_error:
    st.session_state.upload_error = False

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        header = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
        if len(header) != 12:
            st.error("The file must contain exactly 12 columns.")
        else:
            dtypes = {header[11]: str}
            uploaded_file.seek(0)  
            df = pd.read_csv(uploaded_file, dtype=dtypes)
            
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
                    save_path = os.path.join(UPLOAD_FOLDER, "dataFrame.csv")
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.info(f"File saved in: `{save_path}`")
                else:
                    st.error("The last column must contain only string values.")
                    
    except Exception as e:
        st.session_state.upload_error = True
        st.error(f"Error processing the file: {e}")
