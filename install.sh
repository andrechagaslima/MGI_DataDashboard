sudo apt update -qq

if ! command -v git &> /dev/null; then
    DEBIAN_FRONTEND=noninteractive sudo apt install -y git
fi

pip install -r requirements.txt
python -m spacy download pt_core_news_sm