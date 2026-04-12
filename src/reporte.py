import pandas as pd
import io

def generate_excel_report(df, analysis_result):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:

        #  Datos limpios
        df.to_excel(writer, index=False, sheet_name='Datos')

        #  Estadísticas
        stats_df = pd.DataFrame(analysis_result["Estadísticas"]).T
        stats_df.to_excel(writer, sheet_name='Estadisticas')

        #  Correlación
        corr_df = pd.DataFrame(analysis_result["Matriz"])
        corr_df.to_excel(writer, sheet_name='Correlacion')

        #  Insights automáticos
        insights = analysis_result["Insights"]
        insights_df = pd.DataFrame({"Insights": insights})
        insights_df.to_excel(writer, sheet_name='Insights', index=False)

    output.seek(0)
    return output