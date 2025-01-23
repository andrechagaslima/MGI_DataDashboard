
echo "Installing PythonVEnv"
sudo apt install -y python3-venv 

echo "Setting FSLLM"
python3.10 -m venv fsllm

printf "\nexport FSLLMWORKDIR=`dirname $PWD`" >> fsllm/bin/activate

source fsllm/bin/activate

pip install --upgrade pip wheel setuptools

pip install -r llm_requirements.txt

# deactivate