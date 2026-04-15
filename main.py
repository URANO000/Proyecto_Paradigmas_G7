import streamlit as st
import pandas as pd
from src.loader import Loader
from src.cleaner import DataCleaner
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer
import io


def display_insights(insights_list):
   
    if not insights_list:
        st.info("No hay insights disponibles para este dataset.")
        return
    
    # Agrupar insights por categoría
    categories = {}
    current_category = None
    
    for insight in insights_list:
        if insight.startswith("\n") and insight.isupper() or insight.startswith("\n"):
            current_category = insight.strip("\n").strip()
        else:
            if current_category and insight.strip():
                if current_category not in categories:
                    categories[current_category] = []
                categories[current_category].append(insight.strip())
    
    # Mostrar cada categoría en un contenedor con bordes
    for category, insights in categories.items():
        if not category or not insights:
            continue
            
        with st.container(border=True):
            st.markdown(f"**{category}**")
            
            for insight in insights:
                if insight and not insight.startswith("---"):
                    # Limpiar y presentar el insight
                    cleaned_insight = insight.strip("• ").strip()
                    if cleaned_insight and not cleaned_insight.upper() == cleaned_insight:
                        st.markdown(f"• {cleaned_insight}")
                    elif cleaned_insight:
                        st.text(cleaned_insight)


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


def generate_excel_report(df, result):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        # 📊 Datos
        df.to_excel(writer, index=False, sheet_name='Datos')

        # 📉 Estadísticas
        stats_df = df.describe().T
        stats_df.to_excel(writer, sheet_name='Estadisticas')

        # 📈 Correlación
        corr_df = pd.DataFrame(result["Matriz"])
        corr_df.to_excel(writer, sheet_name='Correlacion')

        # 🤖 Insights
        insights = result["Insights Automáticos"]
        insights_df = pd.DataFrame({"Insights": insights})
        insights_df.to_excel(writer, sheet_name='Insights', index=False)

    output.seek(0)
    return output


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
    tab1, tab2, tab3, tab4, tab5= st.tabs([
        "Resumen",
        "Datos",
        "Análisis",
        "Visualizaciones",
        "Reporte"
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

        st.divider()
        st.markdown("### Insights Automáticos")
        display_insights(result["Insights Automáticos"])


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


   
    # TAB 5: REPORTE
   

    with tab5:
        st.subheader("Reporte inteligente")

        st.success("Reporte generado automáticamente listo para descarga")
        
        excel_file = generate_excel_report(df, result)
        st.download_button(
            label="Descargar Excel completo",
            data=excel_file,
            file_name=f"reporte_inteligente_{selected_file}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### Vista previa del reporte")

        # INFO
        st.markdown("#### Información general")
        st.write(f"Filas: {df.shape[0]}")
        st.write(f"Columnas: {df.shape[1]}")

        # STATS
        st.markdown("#### Estadísticas descriptivas")
        stats_df = df.describe().T
        st.dataframe(stats_df, use_container_width=True)

        # CORRELACIÓN
        st.markdown("#### Matriz de correlación")
        corr_df = pd.DataFrame(result["Matriz"])
        st.dataframe(corr_df, use_container_width=True)

        # INSIGHTS
        st.markdown("#### Insights automáticos")
        display_insights(result["Insights Automáticos"])

        # DESCARGA
        st.markdown("### Descargar reporte completo")



else:
    st.info("Sube uno o más archivos para comenzar")