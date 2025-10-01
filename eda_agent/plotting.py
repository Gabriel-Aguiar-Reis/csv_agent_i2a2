import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_graph(df: pd.DataFrame, question: str, tool_type: str = "histogram", **kwargs):
    """
    Gera diferentes tipos de gráficos com base no tool_type.
    tool_type: histogram, boxplot, scatter, heatmap, bar, line, cluster, crosstab
    kwargs: argumentos extras para gráficos específicos
    """
    match tool_type:
        case "histogram":
            num_cols = df.select_dtypes(include='number').columns
            figs = []
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=col, ax=ax)
                ax.set_title(f"Histograma de {col}")
                figs.append(fig)
            if figs:
                return figs if len(figs) > 1 else figs[0]
        case "boxplot":
            fig, ax = plt.subplots()
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                sns.boxplot(data=df[num_cols], ax=ax)
                ax.set_title("Boxplot das variáveis numéricas")
                return fig
        case "scatter":
            fig, ax = plt.subplots()
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) >= 2:
                x = kwargs.get("x", num_cols[0])
                y = kwargs.get("y", num_cols[1])
                sns.scatterplot(data=df, x=x, y=y, ax=ax)
                ax.set_title(f"Scatter plot: {x} vs {y}")
                return fig
        case "heatmap":
            corr = df.corr(numeric_only=True)
            # Heatmap maior para melhor visualização
            n = len(corr)
            fig_width = max(12, 0.7 * n)
            fig_height = max(10, 0.7 * n)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                corr,
                annot=False,
                cmap="coolwarm",
                ax=ax,
                cbar=True,
                square=True,
                linewidths=0.5,
                xticklabels=True,
                yticklabels=True
            )
            # Ajusta os rótulos para evitar sobreposição
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
            fig.tight_layout()
            ax.set_title("Heatmap de Correlação (cores apenas, sem valores)")
            return fig
        case "bar":
            fig, ax = plt.subplots()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                col = cat_cols[0]
                counts = df[col].value_counts().head(10)
                sns.barplot(x=counts.index, y=counts.values, ax=ax)
                ax.set_title(f"Top categorias de {col}")
                ax.set_ylabel("Frequência")
                ax.set_xlabel(col)
                return fig
        case "line":
            fig, ax = plt.subplots()
            date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns
            num_cols = df.select_dtypes(include='number').columns
            if len(date_cols) > 0 and len(num_cols) > 0:
                x = date_cols[0]
                y = num_cols[0]
                sns.lineplot(data=df, x=x, y=y, ax=ax)
                ax.set_title(f"Linha temporal: {y} ao longo de {x}")
                return fig
        case "cluster":
            fig, ax = plt.subplots()
            from sklearn.cluster import KMeans
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) >= 2:
                X = df[num_cols].dropna()
                n_clusters = min(3, len(X))
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                    labels = kmeans.fit_predict(X)
                    sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=labels, palette="tab10", ax=ax)
                    ax.set_title(f"Clusters (KMeans) em {X.columns[0]} vs {X.columns[1]}")
                    return fig
        case "crosstab":
            fig, ax = plt.subplots()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) >= 2:
                ct = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Crosstab: {cat_cols[0]} vs {cat_cols[1]}")
                return fig
        case _:
            pass
    return None
