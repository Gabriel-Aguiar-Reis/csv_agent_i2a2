import os
import streamlit as st
from eda_agent.eda_agent import EDAAgent
from eda_agent.memory import AgentMemory
from eda_agent.figs import fig_to_base64
import pandas as pd
from matplotlib.figure import Figure
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="Agente EDA CSV", layout="wide")
st.title("Agente EDA para CSV - Chat")

# Inicializa memória e agente na sessão
if "memory" not in st.session_state:
    st.session_state.memory = AgentMemory()
if "agent" not in st.session_state:
    st.session_state.agent = EDAAgent(st.session_state.memory, api_key=OPENROUTER_API_KEY)
if "df" not in st.session_state:
    st.session_state.df = None

# Histórico de mensagens (único, como no exemplo)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Envie um CSV ou faça uma pergunta sobre os dados! 👇"}
    ]

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        figs_b64 = message.get("fig", None)
        if figs_b64 is not None:
            if isinstance(figs_b64, list):
                for img_b64 in figs_b64:
                    st.image(img_b64, width=250)
            elif isinstance(figs_b64, str):
                st.image(figs_b64, width=250)

# Entrada do usuário (CSV ou texto)
chat_value = st.chat_input("Digite sua mensagem", accept_file=True, file_type=["csv"], disabled=st.session_state.messages[-1]["role"] != "assistant")

if chat_value:
    # Se for upload de arquivo CSV
    if hasattr(chat_value, "files") and chat_value.files:
        uploaded_file = chat_value.files[0]
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.agent.load_data(st.session_state.df)
        response = f"Arquivo CSV '{uploaded_file.name}' carregado com sucesso!"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Se for texto (pergunta)
    if hasattr(chat_value, "text") and chat_value.text:
        user_text = chat_value.text
        st.session_state.messages.append({"role": "user", "content": user_text})
        # Mensagem temporária de processamento
        processing_msg = {"role": "assistant", "content": "Processando..."}
        st.session_state.messages.append(processing_msg)
        # Atualiza a interface para mostrar imediatamente
        with st.chat_message("user"):
            st.markdown(user_text)
        with st.chat_message("assistant"):
            st.markdown("Processando...")
        # Processa a resposta
        response, fig = st.session_state.agent.answer_question(user_text)
        # Remove a mensagem temporária
        st.session_state.messages = st.session_state.messages[:-1]
        # Exibe e adiciona a resposta real
        with st.chat_message("assistant"):
            st.markdown(response)
            figs_b64 = None
            if fig is not None:
                if isinstance(fig, list):
                    figs_b64 = [fig_to_base64(f) for f in fig if isinstance(f, Figure)]
                    for img_b64 in figs_b64:
                        st.image(img_b64, width=400)
                elif isinstance(fig, Figure):
                    figs_b64 = fig_to_base64(fig)
                    st.image(figs_b64, width=400)
        msg = {"role": "assistant", "content": response}
        if fig is not None:
            msg["fig"] = figs_b64
        st.session_state.messages.append(msg)
