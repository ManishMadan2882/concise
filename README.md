### Install Llama 3.2 on your local system

``curl -fsSL https://ollama.com/install.sh | sh``
Note: For Windows download Ollama Setup manually

``ollama run llama3.2``

## Quickstart

### Create Virtual Env
Make sure the is set to concise/application
`cd application`
#### Windows 
`python -m venv venv`
#### Mac/Linux 
`python3 -m venv venv`



`source venv/bin/activate`

### Install the dependencies

`pip install -r requirements.txt`

## run the application

`streamlit run rag/index.py`

App runs on PORT 8501