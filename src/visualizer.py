import matplotlib.pyplot as plt
import pandas as pd


class DataVisualizer:

    def plot_histograms(self, df, numeric_cols):
        figures = []

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[col].dropna(), bins=20, edgecolor="black")
            ax.set_title(f"Histograma de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            fig.tight_layout()
            figures.append(fig)

        return figures

    def plot_boxplots(self, df, numeric_cols):
        figures = []

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.boxplot(df[col].dropna(), vert=False)
            ax.set_title(f"Boxplot de {col}")
            ax.set_xlabel(col)
            fig.tight_layout()
            figures.append(fig)

        return figures

    def plot_categorical_bars(self, df, categorical_cols, max_categories=10):
        figures = []

        for col in categorical_cols:
            value_counts = df[col].astype(str).value_counts().head(max_categories)

            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Frecuencia de categorías en {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Cantidad")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            figures.append(fig)

        return figures

    def plot_scatter(self, df, numeric_cols):
        figures = []

        if len(numeric_cols) < 2:
            return figures

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_x = numeric_cols[i]
                col_y = numeric_cols[j]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df[col_x], df[col_y], alpha=0.7)
                ax.set_title(f"Dispersión: {col_x} vs {col_y}")
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                fig.tight_layout()
                figures.append(fig)

        return figures

    def plot_correlation_heatmap(self, corr_matrix):
        if corr_matrix.empty:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto")

        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.index)

        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

        ax.set_title("Mapa de calor de correlación")
        fig.colorbar(cax)
        fig.tight_layout()

        return fig

    def plot_cluster_scatter(self, df, numeric_cols):
        if "cluster" not in df.columns or len(numeric_cols) < 2:
            return None

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        fig, ax = plt.subplots(figsize=(8, 5))

        for cluster_id in sorted(df["cluster"].dropna().unique()):
            cluster_data = df[df["cluster"] == cluster_id]
            ax.scatter(
                cluster_data[x_col],
                cluster_data[y_col],
                label=f"Cluster {cluster_id}",
                alpha=0.7
            )

        ax.set_title(f"Clusters detectados: {x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        fig.tight_layout()

        return fig