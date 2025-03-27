# Importação das Bibliotecas necessárias
import os
from dotenv import load_dotenv, find_dotenv # Biblioteca para carregar variáveis de ambiente
from langchain_groq import ChatGroq # Integração do LangChain com Groq
from langchain_community.chat_message_histories import ChatMessageHistory #Permite criar histórico de MSG
from langchain_core.chat_history import BaseChatMessageHistory #Cria uma classe base para histórioc de MSG
from langchain_core.runnables.history import RunnableWithMessageHistory #Permite Gerenciar o histórico de MSG
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Permite criar prompts e MSG
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages #MSG Humanas, sistema e do AI
from langchain_core.runnables import RunnablePassthrough #Permite criar fluxos de execução e reutilizaveis
from operator import itemgetter #Permite a extração de valores de dicionários

# Carregar as variáveis de ambinete do arquivo .env (para proteger as credenciais)
load_dotenv()

# Obter a chave da API do Groq que está armazenada no arquivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o modelo de AI utilizando a API da GROQ
model = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = GROQ_API_KEY
)

# EXEMPLO 01---------------------------------------------------------------------
# Dicionário para armazenar o histórico de mensagens
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recupera ou cria um histório de mensagens para uma determinada sessão
    Isso permite manter o contexto continuo para diferentes usuários e interações.
    """
    if session_id not in store:
        store [session_id] = ChatMessageHistory()
    return store[session_id]

# Criar um gerenciador de histórico que coneta pelo armazenamento de mensagens
with_message_history = RunnableWithMessageHistory(model, get_session_history)

#Configuração da sessão (Identificador único para cada usuário)
config = {"configurable":{"session_id":"chat1"}}

#Exemplo de interação inicial do usuário
response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Julio e eu sou Administrador")],
    config=config
)

# EXEMPLO 02-------------------------------------------------------------------------------------------
# Criação de um prompt template para estrutura a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", " Você é um assistente útil. Responda todas as perguntas  suas habilidades em {language}."),
        MessagesPlaceholder(variable_name="messages") # Permitir adicionar mensagens de forma dinâmica
    ]
)

# Conecta o modelo ao template de prompt
chain = prompt | model

# Exemplo de interação usando o template
chain.invoke({"messages": [HumanMessage(content="Oi, meu nome é Julio")], "language": "Inglês"})

# Gerenciamento da memória do chabot
trimmer = trim_messages(
    max_tokens = 45, # Define um limite máximo de tokens para evitar ultrapassar o consumo de memória
    strategy = "last", # Define a estratégia de corte para remover mensagens antigas
    token_counter = model, # usa o modelo para contar os tokens
    include_system = True, # Inclui mesnagens do sistema no histórico
    allow_partial = False, # Evita que as menssagens sejam cortadas parcialmente
    start_on = "human" # Começa a contagem dos tokens com a mensagem humana
)
# Exemplo de histórico de mensagens
messages = [
    SystemMessage(content="Você é um assistente."),
    HumanMessage(content="Oi, o meu nome é John Wick."),
    AIMessage(content="Oi John, como posso te ajudar hoje?"),
    HumanMessage(content="Eu gosto de sorvete de Pistache.")
]

# Aplicar o limitador de memória ao histórico 
trimmer.invoke(messages)

# Criando um pipeline de execução para otimizar a passagem de informações entre os componentes
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) # Aplica otimização do histórico
    | prompt # Passa a entrada pelo template do prompt
    | model # Passa as entradas pelo modelo
)

# Exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="Qual é o sorvete que eu gosto?")],
        "language": "Inglês"
    }
)

# Exibir a resposta final do modelo
print("Resposta final do modelo", response.content)