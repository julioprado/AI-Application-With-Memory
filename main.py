# Importação das Bibliotecas necessárias
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory #Permite criar histórico de MSG
from langchain_core.chat_history import BaseChatMessageHistory #Cria uma classe base para histórioc de MSG
from langchain_core.runnables.history import RunnableWithMessageHistory #Permite Gerenciar o histórico de MSG
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder #Permite criar prompts e MSG
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages #MSG Humanas, sistema e do AI
from langchain_core.runnables import RunnablePassthrough #Permite criar fluxos de execução e reutilizaveis
from operator import itemgetter #Permite a extração de valores de dicionários

# Carregar as variáveis de ambinete do arquivo .env (para proteger as credenciais)
load_dotenv(find_dotenv())

# Obter a chave da API do Groq que está armazenada no arquivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o modelo de AI utilizando a API da GROQ
model = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = GROQ_API_KEY
)

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

# Exibir a resposta do modelo
print("Resposta do modelo:", response.content)

        