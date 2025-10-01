
import os
from openai import OpenAI
from dotenv import load_dotenv

def query_llm(question: str, analysis):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    prompt = (
        f"Dados analisados: {analysis}\n"
        f"Pergunta: {question}\n"
        "\nINSTRUÇÕES PARA USO DE FERRAMENTAS (TOOLS):\n"
        "Se a resposta exigir um gráfico, ao final da resposta escreva uma linha 'tool:<nome_da_tool>' conforme a lista abaixo.\n"
        "Se não for necessário gráfico, escreva 'tool:none'.\n"
        "\nTOOLS DISPONÍVEIS:\n"
        "- tool:histogram   → Histogramas das variáveis numéricas.\n"
        "- tool:boxplot     → Boxplot das variáveis numéricas.\n"
        "- tool:scatter     → Scatter plot entre duas variáveis numéricas.\n"
        "- tool:heatmap     → Heatmap de correlação entre variáveis numéricas.\n"
        "- tool:bar         → Gráfico de barras das categorias mais frequentes.\n"
        "- tool:line        → Linha temporal (se houver coluna de data).\n"
        "- tool:cluster     → Scatter plot colorido por cluster (KMeans).\n"
        "- tool:crosstab    → Heatmap de tabela cruzada entre duas variáveis categóricas.\n"
        "- tool:none        → Não é necessário gráfico.\n"
        "\nEXEMPLOS:\n"
        "- 'Aqui está a distribuição dos valores.\ntool:histogram'\n"
        "- 'A relação entre as variáveis X e Y está representada abaixo.\ntool:scatter'\n"
        "- 'Não há necessidade de gráfico para responder.\ntool:none'\n"
        "\nSempre escolha a tool mais adequada para ilustrar sua resposta.\n"
        "Resposta: "
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        extra_body={},
        model="x-ai/grok-4-fast:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content
