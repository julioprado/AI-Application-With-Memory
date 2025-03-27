# AI Chatbot com Memória - GROQ API

## ✨ Visão Geral
Este projeto implementa um chatbot com memória utilizando a **GROQ API**. O chatbot é capaz de manter o contexto de conversações ao longo do tempo, armazenando mensagens anteriores e gerenciando eficientemente o histórico.

---

## 🌐 Em Produção
Este chatbot pode ser integrado a aplicativos de mensagens, assistentes virtuais ou serviços de atendimento ao cliente. Ele usa o modelo **Gemma2-9B-IT** para gerar respostas e gerencia eficientemente a memória de conversas.

---

## ℹ️ Sumário
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Glossário](#-glossário)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Explicação dos Blocos de Código](#-explicação-dos-blocos-de-código)
- [Uso](#-uso)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

---

## 📚 Glossário
- **Histórico de Mensagens**: Registro das interações entre o usuário e o chatbot.
- **Chain**: Um pipeline que conecta componentes de processamento para otimizar a geração de respostas.
- **Trim**: Mecanismo para limitar o uso de memória cortando mensagens antigas.
- **Runnable**: Objeto que executa uma tarefa dentro do LangChain.

---

## 🛠️ Tecnologias Utilizadas
- **Python 3.10+**
- **LangChain** (para gestão de histórico e pipelines)
- **GROQ API** (para processamento de linguagem natural)
- **dotenv** (para gerenciamento de variáveis de ambiente)

---

## ♻️ Instalação

### 1️⃣ Clonar o Repositório
```sh
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
```

### 2️⃣ Criar e Ativar o Ambiente Virtual
```sh
    python -m venv .venv
    # No Windows:
    .venv\Scripts\activate
    # No macOS/Linux:
    source .venv/bin/activate
```

### 3️⃣ Instalar as Dependências
```sh
    pip install -r requirements.txt
```

---

## 🔧 Configuração
Crie um arquivo **.env** na raiz do projeto e adicione sua chave da API:
```ini
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💡 Explicação dos Blocos de Código

### 1️⃣ **Importação das Bibliotecas**
Carrega módulos necessários para interação com a API, histórico de mensagens e processamento de prompts.

### 2️⃣ **Configuração do Modelo**
Lê a chave da API e inicializa o modelo da GROQ.
```python
    load_dotenv(find_dotenv())
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
```

### 3️⃣ **Gerenciamento do Histórico**
Permite que o chatbot lembre interações passadas.
```python
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
```

### 4️⃣ **Criação de Prompts**
Estrutura a entrada do chatbot para melhorar a resposta.
```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente útil."),
        MessagesPlaceholder(variable_name="messages")
    ])
```

### 5️⃣ **Gerenciamento de Memória**
Define regras para limitar a quantidade de informações armazenadas.
```python
    trimmer = trim_messages(
        max_tokens=45, strategy="last", token_counter=model,
        include_system=True, allow_partial=False, start_on="human"
    )
```

### 6️⃣ **Pipeline de Processamento**
Cria um fluxo otimizado para processar as mensagens.
```python
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt | model
    )
```

---

## 🔄 Uso

### Exemplo de Interação com Memória
```python
    response = with_message_history.invoke(
        [HumanMessage(content="Oi, meu nome é Julio e eu sou Administrador")],
        config={"configurable": {"session_id": "chat1"}}
    )
    print(response.content)
```

### Exemplo de Uso do Pipeline
```python
    response = chain.invoke(
        {"messages": messages + [HumanMessage(content="Qual é o sorvete que eu gosto?")],}
    )
    print("Resposta final do modelo", response.content)
```

---

## 📚 Contribuição
Contribuições são bem-vindas! Para sugerir melhorias:
1. **Fork** este repositório.
2. Crie um **branch** para sua feature (`git checkout -b feature-nova`).
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova feature'`).
4. Faça um **push** (`git push origin feature-nova`).
5. Abra um **Pull Request**.

---

## ⚖️ Licença
Este projeto está sob a [MIT License](LICENSE).