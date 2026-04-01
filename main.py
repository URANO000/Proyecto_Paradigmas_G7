import streamlit as st
import pandas as pd
from src.loader import Loader
from src.cleaner import DataCleaner
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer



# PAGE CONFIG

st.set_page_config(
    page_title="Sistema inteligente de análisis",
    layout="wide"
)

st.title("Sistema inteligente de análisis")
st.caption("Carga, limpieza, análisis y visualización interactiva")



# SIDEBAR

with st.sidebar:
    st.header("Configuración :gear:")

    uploaded_files = st.file_uploader(
        "Agregar archivos",
        accept_multiple_files=True,
        type=["csv", "xlsx", "json", "parquet", "txt", "html", "feather"]
    )



# CACHED DATA PROCESSING

@st.cache_data
def process_data(files):
    loader = Loader(files)
    df_list = loader.load_dataframeList()

    cleaner = DataCleaner(df_list)
    clean_dfList = cleaner.clean_dataframeList()

    analyzer = DataAnalyzer(clean_dfList)
    results = analyzer.analyze_dataframeList()

    return clean_dfList, results



# MAIN LOGIC

if uploaded_files:

    with st.spinner("Procesando datos..."):
        try:
            clean_dfList, results = process_data(uploaded_files)
        except Exception as e:
            st.error(f"Error en procesamiento: {e}")
            st.stop()

    visualizer = DataVisualizer()


    # FILE SELECTOR

    file_names = [file.name for file in uploaded_files]
    selected_file = st.selectbox("Seleccionar archivo", file_names)

    idx = file_names.index(selected_file)

    df = clean_dfList[idx]
    result = results[idx]

    numeric_cols = list(result["Columnas Numéricas"])
    categorical_cols = list(result["Columnas Categóricas"])

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen",
        "Datos",
        "Análisis",
        "Visualizaciones"
    ])

    # TAB 1: RESUMEN

    with tab1:
        st.subheader("Resumen del dataset")

        col1, col2 = st.columns(2)
        col1.metric("Columnas numéricas", len(numeric_cols))
        col2.metric("Columnas categóricas", len(categorical_cols))

        st.markdown("### Columnas")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Numéricas")
            st.write(numeric_cols)

        with col2:
            st.write("Categóricas")
            st.write(categorical_cols)

        st.markdown("### Insights automáticos")
        st.write(result["Insights Automáticos"])


    # TAB 2: DATOS

    with tab2:
        st.subheader("DataFrame")

        with st.expander("Ver datos completos"):
            st.dataframe(df, use_container_width=True)


    # TAB 3: ANÁLISIS

    with tab3:
        st.subheader("Análisis estadístico")

        with st.expander("Matriz de correlación"):
            st.dataframe(result["Matriz"], use_container_width=True)


    # TAB 4: VISUALIZACIONES

    with tab4:
        st.subheader("Visualización interactiva")

        # -------- NUMERIC --------
        if numeric_cols:
            st.markdown("### Variables numéricas")

            selected_num_col = st.selectbox(
                "Seleccionar columna",
                numeric_cols,
                key="num_col"
            )

            plot_type = st.radio(
                "Tipo de gráfico",
                ["Histograma", "Boxplot"],
                horizontal=True
            )

            if plot_type == "Histograma":
                fig = visualizer.plot_histograms(df, [selected_num_col])[0]
                st.pyplot(fig)

            elif plot_type == "Boxplot":
                fig = visualizer.plot_boxplots(df, [selected_num_col])[0]
                st.pyplot(fig)

        # -------- SCATTER --------
        if len(numeric_cols) >= 2:
            st.markdown("### Relación entre variables")

            col1, col2 = st.columns(2)

            x_col = col1.selectbox("Eje X", numeric_cols, key="x")
            y_col = col2.selectbox("Eje Y", numeric_cols, key="y")

            fig = visualizer.plot_scatter(df, [x_col, y_col])[0]
            st.pyplot(fig)

            if st.checkbox("Mostrar clustering"):
                cluster_fig = visualizer.plot_cluster_scatter(df, numeric_cols)
                if cluster_fig:
                    st.pyplot(cluster_fig)

            if st.checkbox("Mostrar heatmap de correlación"):
                corr_df = pd.DataFrame(result["Matriz"])
                heatmap_fig = visualizer.plot_correlation_heatmap(corr_df)
                if heatmap_fig:
                    st.pyplot(heatmap_fig)

        # -------- CATEGORICAL --------
        if categorical_cols:
            st.markdown("### Variables categóricas")

            selected_cat_col = st.selectbox(
                "Seleccionar categoría",
                categorical_cols[:5],
                key="cat_col"
            )

            fig = visualizer.plot_categorical_bars(df, [selected_cat_col])[0]
            st.pyplot(fig)

else:
    st.info("Sube uno o más archivos para comenzar")