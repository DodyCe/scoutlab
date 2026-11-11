"""Intermediaria HTTP entre el navegador y main_api en el puerto local 8001."""
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(
        base_url='http://127.0.0.1:8001', timeout=10.0, trust_env=False
    ) as cliente:
        app.state.cliente = cliente
        yield


app = FastAPI(
    title='ScoutLab intermediaria',
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get('/')
@app.post('/predict')
async def comunicar_con_main(request: Request):
    # Solo se reenvían las dos rutas declaradas a un destino local fijo.
    try:
        respuesta = await request.app.state.cliente.request(
            method=request.method,
            url=request.url.path,
            content=await request.body(),
            headers={'Content-Type': request.headers.get('Content-Type', 'application/json')},
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={'mensaje': 'La API principal tardó demasiado en responder.'},
        )
    except httpx.RequestError:
        return JSONResponse(
            status_code=503,
            content={'mensaje': 'No se pudo conectar con la API principal.'},
        )

    return Response(
        content=respuesta.content,
        status_code=respuesta.status_code,
        headers={'Content-Type': respuesta.headers.get('Content-Type', 'application/json')},
    )
