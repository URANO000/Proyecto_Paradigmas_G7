# Método main
import pandas as pd
from src.loader import Loader
from src.cleaner import DataCleaner

# Dataset lista, para pruebas nada más
datasetList = ['data/data.csv', 'data/data.json']

# Prueba de loader
loader = Loader(datasetList)
df = loader.load_dataframeList()

# Prueba de cleaner
cleaner = DataCleaner(df)
clean_dfList = cleaner.clean_dataframeList()

# Ver los DataFrames limpios
for df in clean_dfList:
    print(df.to_string())