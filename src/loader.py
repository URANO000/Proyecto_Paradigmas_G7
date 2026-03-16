# Este módulo tiene como propósito el cargar todos los datasets 
# (sea uno o muchos, y de diferentes extensiones)
import pandas as pd

class Loader:

    def __init__(self, datasetList):
        self.datasetList = datasetList

    def load_dataframeList(self):
            dataframeList = []
            # Un loop para iterar por archivo dentro de la lista de archivos
            for dataset in self.datasetList:
                df = self.read_dataset(dataset)
                dataframeList.append(df)
            
            # Retorna la lista de DataFrames
            return dataframeList


    def read_dataset(self, dataset):
        # Método que retorna el DataFrame apropiado por item
            if dataset.endswith('.csv'):
                return pd.read_csv(dataset)
            elif dataset.endswith('.xlsx'):
                return pd.read_excel(dataset)
            elif dataset.endswith('.json'):
                return pd.read_json(dataset)
            elif dataset.endswith('.parquet'):
                return pd.read_parquet(dataset)
            elif dataset.endswith('.txt') | dataset.endswith('.fwf'):
                return pd.read_fwf(dataset)
            elif dataset.endswith('.html'):
                return pd.read_html(dataset)
            elif dataset.endswith('.feather'):
                return pd.read_feather(dataset)
            elif dataset.endswith('.h5') | dataset.endswith('.hdf'):
                return pd.read_hdf(dataset)
            # Si no es ninguna de estas extensiones, entonces error!
            else:
                return ValueError("El o los archivos tienen una extensión de archivo no compatible.")
        