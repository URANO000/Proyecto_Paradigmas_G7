import streamlit as st
import pandas as pd
from src.loader import Loader
from src.cleaner import DataCleaner
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer

st.title("Sistema inteligente de análisis")

uploaded_files = st.file_uploader("Agregar archivos", accept_multiple_files=True, type=["csv", "xlsx", "json", "parquet", "txt", "html", "feather"])

datasetList = []

# lista de archivos
if uploaded_files:

    loader = Loader(uploaded_files)

    try:
        df_list = loader.load_dataframeList()
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        st.stop()

    cleaner = DataCleaner(df_list)

    try:
        clean_dfList = cleaner.clean_dataframeList()
    except Exception as e:
        st.error(f"Error limpiando datos: {e}")
        st.stop()

    analyzer = DataAnalyzer(clean_dfList)
    visualizer = DataVisualizer()

    try:
        results = analyzer.analyze_dataframeList()
    except Exception as e:
        st.error(f"Error en análisis: {e}")
        st.stop()


    # Mostrar resultados por archivo
    for file, df, result in zip(uploaded_files, clean_dfList, results):

        st.header(f"Archivo: {file.name}")

        st.subheader("DataFrame limpio")
        st.dataframe(df)

        st.subheader("Columnas")
        st.write("Numéricas:", result["Columnas Numericas"])
        st.write("Categóricas:", result["Columnas Categoricas"])

        st.subheader("Insights")
        st.write(result["Insights Automaticos"])

        st.subheader("Matriz de correlación")
        st.dataframe(result["Matriz"])

        st.subheader("Visualizaciones")

        numeric_cols = list(result["Columnas Numericas"])
        categorical_cols = list(result["Columnas Categoricas"])

        if numeric_cols:
            st.markdown("### Histogramas")
            for fig in visualizer.plot_histograms(df, numeric_cols):
                st.pyplot(fig)

            st.markdown("### Boxplots")
            for fig in visualizer.plot_boxplots(df, numeric_cols):
                st.pyplot(fig)

        if len(numeric_cols) >= 2:
            st.markdown("### Diagramas de dispersión")
            for fig in visualizer.plot_scatter(df, numeric_cols):
                st.pyplot(fig)

            st.markdown("### Heatmap de correlación")
            corr_df = pd.DataFrame(result["Matriz"])
            heatmap_fig = visualizer.plot_correlation_heatmap(corr_df)
            if heatmap_fig is not None:
                st.pyplot(heatmap_fig)

            st.markdown("### Visualización de clusters")
            cluster_fig = visualizer.plot_cluster_scatter(df, numeric_cols)
            if cluster_fig is not None:
                st.pyplot(cluster_fig)

        if categorical_cols:
            st.markdown("### Gráficos de barras")
            for fig in visualizer.plot_categorical_bars(df, categorical_cols[:3]):
                st.pyplot(fig)