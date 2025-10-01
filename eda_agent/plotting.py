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
            MAX_VARS = 10
            corr = df.corr(numeric_only=True)
            num_vars = corr.shape[0]
            figs = []
            if num_vars > MAX_VARS:
                # Divide as variáveis em grupos de até MAX_VARS (por variância)
                variancias = df.var(numeric_only=True).sort_values(ascending=False)
                var_names = list(variancias.index)
                for i in range(0, len(var_names), MAX_VARS):
                    group = var_names[i:i+MAX_VARS]
                    corr_group = corr.loc[group, group]
                    fig, ax = plt.subplots()
                    sns.heatmap(corr_group, annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title(f"Heatmap de Correlação ({i+1}-{i+len(group)})\n⚠️ Por limitações visuais, o máximo de variáveis por imagem é {MAX_VARS}.")
                    figs.append(fig)
                return figs if len(figs) > 1 else figs[0]
            else:
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Heatmap de Correlação")
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
