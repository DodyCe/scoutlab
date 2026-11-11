"""API local de inferencia: no se conecta a MySQL ni reentrena al consultar."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

BASE = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    app.state.paquete = joblib.load(BASE/'models/modelo_scouting_rf.joblib')
    yield


app = FastAPI(title='ScoutLab', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class Jugador(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    jugador: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=100)]
    posicion: Literal['Attack','Midfield','Defender','Goalkeeper']
    # Rangos de adultos revisados en la base el 08/11/2026.
    # Para fijar la estatura mínima se excluyeron los valores atípicos de 17 a 19 cm.
    estatura_cm: Annotated[float, Field(ge=155, le=210)]
    edad: Annotated[int, Field(ge=18, le=56, strict=True)]


class Prediccion(BaseModel):
    jugador: str
    clasificacion: Literal['Élite','Estándar']
    probabilidad_elite: float
    probabilidad_estandar: float


@app.get('/', include_in_schema=False)
def inicio():
    return FileResponse(BASE/'index.html')


@app.get('/health')
def estado():
    return {'estado':'listo', 'modelo':'Random Forest', 'arboles':100}


@app.post('/predict', response_model=Prediccion)
def predecir(entrada: Jugador, request: Request):
    paquete = request.app.state.paquete
    modelo = paquete['modelo']
    # El nombre identifica al candidato y no se incorpora a sus características.
    datos = pd.DataFrame([{
        'edad':entrada.edad, 'estatura_cm':entrada.estatura_cm, 'posicion':entrada.posicion,
    }])
    # Reutilizar las medianas y todas las categorías aprendidas al entrenar.
    datos['estatura_cm'] = datos['estatura_cm'].fillna(
        datos['posicion'].map(paquete['medianas_estatura'])
    ).fillna(paquete['mediana_global_estatura'])
    datos['edad'] = datos['edad'].fillna(paquete['mediana_global_edad'])
    datos['posicion'] = pd.Categorical(
        datos['posicion'], categories=paquete['categorias_posicion']
    )
    dummies = pd.get_dummies(datos[['posicion']], drop_first=True, dtype=int)
    X_nuevo = pd.concat([datos[['edad', 'estatura_cm']], dummies], axis=1)
    X_nuevo = X_nuevo.reindex(columns=paquete['columnas_X'], fill_value=0)
    probabilidades = modelo.predict_proba(X_nuevo.to_numpy())[0]
    indice = list(modelo.classes_).index(1)
    probabilidad = float(probabilidades[indice])
    clase = int(modelo.predict(X_nuevo.to_numpy())[0])
    return Prediccion(
        jugador=entrada.jugador, clasificacion='Élite' if clase==1 else 'Estándar',
        probabilidad_elite=round(probabilidad*100,2),
        probabilidad_estandar=round((1-probabilidad)*100,2),
    )


@app.exception_handler(Exception)
async def error_interno(request, exc):
    logging.getLogger('scoutlab').exception('Error de inferencia', exc_info=exc)
    return JSONResponse(status_code=500, content={'mensaje':'No fue posible completar el análisis. Intenta de nuevo.'})
