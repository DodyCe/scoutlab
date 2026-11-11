"""Conector compartido de las prácticas: configuración exclusivamente del entorno."""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, URL

carpeta = Path(__file__).resolve().parents[2]
archivo_env = carpeta / '.env'
if not archivo_env.is_file():
    archivo_env = carpeta.parent / '.env'
load_dotenv(archivo_env)


class MySQLConnector:
    def __init__(self):
        requeridas = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
        if any(os.getenv(clave) is None for clave in requeridas):
            raise ValueError('Configure las variables DB_* en el .env del proyecto compartido.')
        url = URL.create(
            'mysql+pymysql', username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'), host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', '3306')), database=os.getenv('DB_NAME'),
        )
        self.engine = create_engine(url, connect_args={'connect_timeout': 10})

    def extract_data(self, query):
        with self.engine.connect() as connection:
            return pd.read_sql_query(text(query), connection)
