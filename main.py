# Método main
import pandas as pd
from src.loader import Loader
from src.cleaner import DataCleaner
from src.analyzer import DataAnalyzer

# Dataset lista, para pruebas nada más
datasetList = ['data/data.csv']

# Prueba de loader
loader = Loader(datasetList)
df = loader.load_dataframeList()

# Prueba de cleaner
cleaner = DataCleaner(df)
clean_dfList = cleaner.clean_dataframeList()

#Probar si el análisis funciona bien
print("\n-----------Dataframes Limpios-----------")
for df in clean_dfList:
    print(df.to_string())

analyzer = DataAnalyzer(clean_dfList)
analysis_results = analyzer.analyze_dataframeList()

print("\n-----------Resultados del Análisis-----------")
for result in analysis_results:
    for key, value in result.items():
        print(f"\n  {key}:")
        print(value)
