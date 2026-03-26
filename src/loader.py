import pandas as pd
import os

class Loader:

    def __init__(self, datasetList):
        self.datasetList = datasetList

    def load_dataframeList(self):
        dataframeList = []

        for dataset in self.datasetList:
            df = self.read_dataset(dataset)
            dataframeList.append(df)

        return dataframeList

    def read_dataset(self, dataset):
        """
        DataSet puede ser:
        - str (file path)
        - UploadedFile (Streamlit)
        """

        # Caso 1 - Archivo en memoria
        if hasattr(dataset, "name") and hasattr(dataset, "read"):
            return self._read_from_uploaded_file(dataset)

        # Caso 2 - Archivo agregado
        elif isinstance(dataset, str):
            return self._read_from_path(dataset)

        else:
            raise TypeError(f"Tipo de dataset no soportado: {type(dataset)}")
        
    # Método para leer directo del path
    def _read_from_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        if path.endswith('.csv'):
            return pd.read_csv(path)

        elif path.endswith('.xlsx'):
            return pd.read_excel(path)

        elif path.endswith('.json'):
            return pd.read_json(path)

        elif path.endswith('.parquet'):
            return pd.read_parquet(path)

        elif path.endswith('.txt') or path.endswith('.fwf'):
            return pd.read_fwf(path)

        elif path.endswith('.html'):
            return pd.read_html(path)[0]  # pandas retorna una lista

        elif path.endswith('.feather'):
            return pd.read_feather(path)

        elif path.endswith('.h5') or path.endswith('.hdf'):
            return pd.read_hdf(path)

        else:
            raise ValueError(f"Extensión no compatible: {path}")


    # Función para leer de un obj de streamlit
    def _read_from_uploaded_file(self, file):
        name = file.name.lower()

        if name.endswith('.csv'):
            return pd.read_csv(file)

        elif name.endswith('.xlsx'):
            return pd.read_excel(file)

        elif name.endswith('.json'):
            return pd.read_json(file)

        elif name.endswith('.parquet'):
            return pd.read_parquet(file)

        elif name.endswith('.txt') or name.endswith('.fwf'):
            return pd.read_fwf(file)

        elif name.endswith('.html'):
            return pd.read_html(file)[0]

        elif name.endswith('.feather'):
            return pd.read_feather(file)

        else:
            raise ValueError(f"Extensión no compatible: {name}")