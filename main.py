# fleet_api/main.py — CORRIGÉ (compatible avec database.py existant)
# Utilise db.init() / db.close() au lieu de init_db / close_db

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT, API_RELOAD, DEVICE, PROJECT_ROOT
from ai_engine import init_engine
from database import db   # ✅ db.init() / db.close() — pas init_db/close_db

from routers.fleet           import router as fleet_router
from routers.auth_router     import router as auth_router
from routers.location_router import router as location_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Fleet AI API — démarrage")
    logger.info(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    logger.info(f"  DEVICE       : {DEVICE}")
    logger.info("=" * 60)

    await db.init()                                    # ✅ SQLite
    engine = init_engine(project_root=PROJECT_ROOT, device=DEVICE)
    if engine.ready:
        logger.info("  Moteur IA prêt — API disponible")
    else:
        logger.warning("  Moteur IA partiellement chargé")

    yield

    await db.close()
    logger.info("  Fleet AI API — arrêt propre")

app = FastAPI(
    title="Fleet AI API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fleet_router)
app.include_router(auth_router)
app.include_router(location_router)

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Fleet AI API", "version": "2.0.0", "status": "running",
        "docs": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info",
    )