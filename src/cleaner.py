# Este módulo tiene como propósito el limpiar los datasets

class DataCleaner:
    def __init__(self, dataframeList):
        self.dataframeList = dataframeList

    def clean_dataframeList(self):
        # Método para limpiar una lista de DataFrames
        clean_dataframeList = []
        # Un loop para iterar por cada DataFrame (df) de la df list, que se limpien y añadan a la lista de dfs limpios
        for df in self.dataframeList:
            clean_df = self.clean_dataframe(df)
            clean_dataframeList.append(clean_df)
        return clean_dataframeList

    def clean_dataframe(self, df):
        # Método que limpia un dataframe
        df.info() # Nos muestra el número de filas, columnas, tipos de datos y memoria usada
        df.describe(include='all') # Nos muestra estadísticas descriptivas de las columnas

        # Normalizar nombres de las columnas
        df.columns = (
            df.columns
            .str.strip() # Elimina espacios en blanco al inicio y al final del nombre
            .str.lower() # Convierte el nombre a minúsculas
            .str.replace(" ", "_") # Espacios se reemplazan por guiones bajos
        )

        # Eliminar filas duplicadas
        clean_df = df.drop_duplicates()

        # ELiminar valores nulos (exclusivamente)
        clean_df.dropna(axis='columns', how='all', inplace=True)
        clean_df.dropna(axis='index', how='all', inplace=True)

        # Conversión de datos a tipo inferido (por ejemplo, si los datos numéricos están guardados como str)
        clean_df = clean_df.convert_dtypes()

        #  Reemplazar valores nulos por 'Desconocido' si es de tipo categórico, y por la mediana si es numérico
        for column in clean_df.columns:
            if clean_df[column].dtype == "object":
                clean_df[column] = clean_df[column].fillna("Desconocido")
            else:
                clean_df[column] = clean_df[column].fillna(clean_df[column].median())

        return clean_df
