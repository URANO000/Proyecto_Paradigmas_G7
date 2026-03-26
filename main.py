import streamlit as st
from src.loader import Loader
from src.cleaner import DataCleaner
from src.analyzer import DataAnalyzer

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