import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

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
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
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
                    relation_type = "positiva" if corr_value > 0 else "negativa"
                    strong_relations.append(f"Existe una fuerte correlación {relation_type} entre '{col1}' y '{col2}' ({corr_value:.2f})")

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
                    insights.append(f"En '{col}': la categoría '{top_category}' concentra {percentage:.1f}% de los registros, indicando una distribución desbalanceada")

        return insights
    
    #Este método nos da la variable que más variabilidad tiene por medio de un rango de valores
    def range_analysis(self, df, numeric_cols):

        insights = []

        for col in numeric_cols:
            
            value_range = df[col].max() - df[col].min()

            if value_range > df[col].mean():
                variance_pct = (value_range / df[col].mean()) * 100 if df[col].mean() != 0 else 0
                insights.append(f"'{col}' muestra variabilidad significativa con un rango de {value_range:.2f} ({variance_pct:.0f}% respecto a la media)")
        
        return insights
    
    #Este método aplica clustering K-Means para agrupar registros automáticamente
    def clustering_kmeans(self, df, numeric_cols, k=3):
        
        if len(numeric_cols) == 0:
            return df, None, []
        
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = model.fit_predict(scaled_data)
            
            df['cluster'] = clusters
            
            cluster_insights = [f"Se identificaron {k} grupos principales mediante clustering K-Means"]
            
            # Información sobre tamaño de clusters
            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            for cluster_id, size in cluster_sizes.items():
                percentage = (size / len(df)) * 100
                cluster_insights.append(f"Grupo {cluster_id}: {size} registros ({percentage:.1f}% del total)")
            
            return df, model, cluster_insights
        except Exception as e:
            return df, None, [f"Error en clustering: {str(e)}"]
    
    #Este método detecta outliers utilizando el método del Rango Intercuartílico (IQR)
    def detect_outliers_iqr(self, df, numeric_cols):
        
        outlier_insights = []
        outlier_details = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df)) * 100
                outlier_details[col] = {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                outlier_insights.append(f"'{col}' contiene {len(outliers)} valores atípicos ({outlier_percentage:.1f}%) fuera de rangos esperados")
        
        return outlier_details, outlier_insights
    
    def detect_anomalies_isolation_forest(self, df, numeric_cols, contamination=0.1):
        
        if len(numeric_cols) < 2:
            return {}, []
        
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            
            anomalies = (anomaly_labels == -1).sum()
            anomaly_percentage = (anomalies / len(df)) * 100
            
            anomaly_insights = []
            if anomalies > 0:
                anomaly_insights.append(f"Se detectaron {anomalies} anomalías ({anomaly_percentage:.1f}%) mediante análisis de comportamiento atípico")
            
            return {
                'model': iso_forest,
                'anomaly_count': anomalies,
                'anomaly_percentage': anomaly_percentage,
                'labels': anomaly_labels
            }, anomaly_insights
        except Exception as e:
            return {}, [f"Error en análisis de anomalías: {str(e)}"]
    
    def feature_importance_analysis(self, df, numeric_cols, corr_matrix):
        
        if len(numeric_cols) < 2:
            return {}, []
        
        try:
            feature_scores = {}
            
            # Scoring basado en correlacion + varianza
            for col in numeric_cols:
                corr_score = abs(corr_matrix[col].sum() - 1) / (len(numeric_cols) - 1)
                
                variance_score = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                
                feature_scores[col] = (corr_score * 0.6) + (variance_score * 0.4)
            
            # Ranking de importancia
            ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            importance_insights = []
            importance_insights.append("Ranking de importancia predictiva de características:")
            
            for idx, (feature, score) in enumerate(ranked_features, 1):
                percentage_importance = (score / max(feature_scores.values())) * 100
                importance_insights.append(f"{idx}. '{feature}': {percentage_importance:.0f}% de influencia en el modelo")
            
            return feature_scores, importance_insights
        except Exception as e:
            return {}, [f"Error en análisis de características: {str(e)}"]
    
    def predictive_analysis(self, df, numeric_cols, corr_matrix):
        
        if len(numeric_cols) < 2:
            return []
        
        try:
            predictions_insights = []
            
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if abs(corr_value) > 0.7:
                        strong_correlations.append((col1, col2, corr_value))
            
            if strong_correlations:
                predictions_insights.append("Análisis predictivo basado en relaciones lineales:")
                
                for col1, col2, corr_value in strong_correlations[:3]:
                    try:
                        X = df[[col1]].values
                        y = df[[col2]].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        r2_score = model.score(X, y)
                        direction = "aumentará" if model.coef_[0][0] > 0 else "disminuirá"
                        precision = f"{r2_score:.0%}"
                        predictions_insights.append(f"Si '{col1}' varía, '{col2}' {direction} (precisión predictiva: {precision})")
                    except:
                        pass
            else:
                predictions_insights.append("No se encontraron relaciones lineales fuertes (> 0.7) entre las variables numéricas para análisis predictivo")
            
            return predictions_insights
        except Exception as e:
            return [f"Error en predicción: {str(e)}"]
    
    #Este método genera insights automáticos consolidados a partir de todos los análisis
    def generate_auto_insights(self, stats, strong_relations, outlier_insights, cluster_insights, dominant_cat, range_var, iso_insights, feature_insights, prediction_insights):
        
        consolidated_insights = []
        
        if strong_relations:
            consolidated_insights.append("\nCORRELACIONES SIGNIFICATIVAS")
            for relation in strong_relations:
                consolidated_insights.append(f"{relation}")
        
        if outlier_insights:
            consolidated_insights.append("\nVALORES ATÍPICOS DETECTADOS")
            for insight in outlier_insights:
                consolidated_insights.append(f"{insight}")
        
        if iso_insights:
            consolidated_insights.append("\nDETECCIÓN DE ANOMALÍAS")
            for insight in iso_insights:
                consolidated_insights.append(f"{insight}")
        
        if dominant_cat:
            consolidated_insights.append("\nDISTRIBUCIÓN DE VARIABLES CATEGÓRICAS")
            for insight in dominant_cat:
                consolidated_insights.append(f"{insight}")
        
        if range_var:
            consolidated_insights.append("\nVARIABLES CON ALTA VARIABILIDAD")
            for insight in range_var:
                consolidated_insights.append(f"{insight}")
        
        if feature_insights:
            consolidated_insights.append("\nIMPORTANCIA DE CARACTERÍSTICAS")
            for insight in feature_insights:
                consolidated_insights.append(insight)
        
        if prediction_insights:
            # Solo agregar si tiene más de un elemento o si el elemento es informativo
            if len(prediction_insights) > 1 or (len(prediction_insights) == 1 and not prediction_insights[0].startswith("No se encontraron")):
                consolidated_insights.append("\nANÁLISIS PREDICTIVO")
                for insight in prediction_insights:
                    consolidated_insights.append(insight)
            elif len(prediction_insights) == 1 and prediction_insights[0].startswith("No se encontraron"):
                # Mostrar el mensaje informativo en la sección predictiva
                consolidated_insights.append("\nANÁLISIS PREDICTIVO")
                consolidated_insights.append(prediction_insights[0])
        
        if cluster_insights:
            consolidated_insights.append("\nANÁLISIS DE AGRUPACIONES")
            for insight in cluster_insights:
                consolidated_insights.append(f"{insight}")
        
        return consolidated_insights

    #Este método se encarga de mostrar en una sola función, todos los métodos creados
    def analyze_dataframe(self, df):

        numeric_cols, categorical_cols, boolean_cols = self.detect_types(df)

        stats = self.descriptive_statistics(df, numeric_cols)

        corr_matrix, strong_relations = self.correlation_analysis(df, numeric_cols)

        dominant = self.dominant_categorical(df, categorical_cols)

        range_variable = self.range_analysis(df, numeric_cols)
        
        df_clustered, kmeans_model, cluster_insights = self.clustering_kmeans(df, numeric_cols, k=3)
        outlier_details, outlier_insights = self.detect_outliers_iqr(df_clustered, numeric_cols)
        iso_details, iso_insights = self.detect_anomalies_isolation_forest(df_clustered, numeric_cols, contamination=0.1)
        feature_scores, feature_insights = self.feature_importance_analysis(df_clustered, numeric_cols, corr_matrix)
        prediction_insights = self.predictive_analysis(df_clustered, numeric_cols, corr_matrix)
        consolidated_insights = self.generate_auto_insights(
            stats, strong_relations, outlier_insights, 
            cluster_insights, dominant, range_variable,
            iso_insights, feature_insights, prediction_insights
        )

        return {
            "Columnas Numéricas" : numeric_cols,
            "Columnas Categóricas" : categorical_cols,
            "Estadísticas" : stats,
            "Correlaciones" : strong_relations,
            "Variable Dominante" : dominant,
            "Alta Variabilidad" : range_variable,
            "Matriz" : corr_matrix.to_dict(),
            "Clustering K-Means" : cluster_insights,
            "Outliers (IQR)" : outlier_details,
            "Anomalías (Isolation Forest)" : iso_details,
            "Feature Importance" : feature_scores,
            "Insights Automáticos" : consolidated_insights
        }
    


