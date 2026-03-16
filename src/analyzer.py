import pandas as pd
import numpy as np

class DataAnalyzer:

    def __init__(self, dataframeList):
        self.dataframeList = dataframeList
    
    #Este método lo que hace es analizar todos los datasets
    def analyze_dataframeList(self):
        results = []

        for df in self.dataframeList:
            result = self.analyze_dataframe(df)
            results.append(result)
        
        return results
    
    # Este método se va a encargar de detectar los tipos de variables de los dataframes
    def detect_types(self, df):

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include='object').columns
        boolean_cols = df.select_dtypes(include='bool').columns

        return numeric_cols, categorical_cols, boolean_cols
    
    #Este método se encargará de que a cada columna que tenga números de un dataframe, le calculará operaciones básicas de la estadística como el promedio, mediana, mínimo, máximo y la desviación estándar
    def descriptive_statistics(self, df, numeric_cols):

        stats = {} #diccionario

        for col in numeric_cols:
            stats[col] = {
                "promedio" : df[col].mean(),
                "mediana" : df[col].median(),
                "std" : df[col].std(),
                "min" : df[col].min(),
                "max" : df[col].max()
            }

        return stats
    
    # Este método va a crear una matriz de correlación, y va a comparar columna con la otra para ver si tienen una relación fuerte
    def correlation_analysis(self, df, numeric_cols):

        corr_matrix = df[numeric_cols].corr()

        strong_relations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):

                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]

                corr_value = corr_matrix.loc[col1, col2]
                
                if abs(corr_value) > 0.75:
                    strong_relations.append(f"Hay una fuerte correlación entre {col1} y {col2}: {corr_value:.2f}")

        return corr_matrix, strong_relations
    
    #Este método nos dará la variable más "dominante" del dataset y acá se puede dar el dato en un porcentaje 
    def dominant_categorical(self, df, categorical_cols):

        insights = []

        for col in categorical_cols:
            distribution = df[col].value_counts(normalize=True)

            if len(distribution) > 0:
                top_category = distribution.index[0]
                percentage = distribution.iloc[0] * 100

                if percentage > 40:
                    insights.append(f"En la variable '{col}', la categoría '{top_category}' domina con {percentage:.2f}% de los registros")

        return insights
    
    #Este método nos da la variable que más variabilidad tiene por medio de un rango de valores
    def range_analysis(self, df, numeric_cols):

        insights = []

        for col in numeric_cols:
            
            value_range = df[col].max() - df[col].min()

            if value_range > df[col].mean():
                insights.append(f"La variable '{col}' tiene alta variedad en sus valores")
        
        return insights

    #Este método se encarga de mostrar en una sola función, todos los métodos creados
    def analyze_dataframe(self, df):

        numeric_cols, categorical_cols, boolean_cols = self.detect_types(df)

        stats = self.descriptive_statistics(df, numeric_cols)

        corr_matrix, strong_relations = self.correlation_analysis(df, numeric_cols)

        dominant = self.dominant_categorical(df, categorical_cols)

        range_variable = self.range_analysis(df, numeric_cols)

        return {
            "Columnas Numéricas" : numeric_cols,
            "Columnas Categoricas" : categorical_cols,
            "Estadísticas" : stats,
            "Correlaciones" : strong_relations,
            "Variable Dominante" : dominant,
            "Alta Variabilidad" : range_variable,
            "Matriz" : corr_matrix 
        }
    



