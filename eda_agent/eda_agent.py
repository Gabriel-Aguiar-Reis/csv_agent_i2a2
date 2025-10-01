import pandas as pd
from .memory import AgentMemory
from eda_agent.analysis import analyze_data
from eda_agent.plotting import generate_graph
from eda_agent.llm import query_llm


class EDAAgent:
    def __init__(self, memory: AgentMemory, api_key: str | None):
        self.memory = memory
        self.df = None
        self.api_key = api_key

    def load_data(self, df: pd.DataFrame):
        self.df = df
        self.memory.add_conclusion("Dados carregados: {} linhas, {} colunas.".format(len(df), len(df.columns)))

    def answer_question(self, question: str):
        if self.df is None:
            return "Nenhum dado carregado.", None
        analysis = analyze_data(self.df)
        print("API Key being used:", self.api_key)  # Debug line to check API key
        llm_response = query_llm(question, analysis, api_key=self.api_key)
        fig = None
        # Lógica de tooling: verifica se o LLM pediu gráfico
        tool = None
        lines = llm_response.strip().splitlines() if llm_response else []
        for line in reversed(lines):
            if line.startswith("tool:"):
                tool = line.replace("tool:", "").strip()
                break
        # Só gera gráfico se o LLM pedir
        if tool and tool != "none":
            try:
                fig = generate_graph(self.df, question, tool_type=tool)
            except Exception:
                fig = None
            # Remove instrução do texto final
            llm_response = "\n".join([line for line in lines if not line.startswith("tool:")]).strip()
        elif tool == "none":
            llm_response = "\n".join([line for line in lines if not line.startswith("tool:")]).strip()
        self.memory.add_conclusion(llm_response if llm_response else "Nenhuma resposta gerada pela LLM.")
        return llm_response, fig
