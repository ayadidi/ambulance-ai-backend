import os
import urllib
from pathlib import Path

# ── Chemin racine du projet ──────────────────────────────────────────────────
PROJECT_ROOT: str = os.getenv(
    "FLEET_PROJECT_ROOT",
    r"C:\Users\diaya\les cours\ensiasd\stage\projet"
)

# ── Device PyTorch ────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("FLEET_DEVICE", "cpu")

# ── Serveur ───────────────────────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"  
API_PORT: int = 8000
API_RELOAD: bool = True       

# ── CORS (Requis par main.py) ─────────────────────────────────────────────────
CORS_ORIGINS: list = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://10.0.2.2",          # Émulateur Android Studio
    "http://127.0.0.1",
]

# ── Liaison Réelle SQL Server (bd_ParcAuto via Authentification Windows) ──────
_params = urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=bd_ParcAuto;"
    "Trusted_Connection=yes;"
)
SQL_SERVER_URL = f"mssql+pyodbc:///?odbc_connect={_params}"

# ── Seuils d'alerte ───────────────────────────────────────────────────────────
CRITICITE_SEUIL_CRITIQUE: float = 0.75   
CRITICITE_SEUIL_HAUTE: float    = 0.50
CACHE_TTL_SECONDS: int = 3600 * 24  # 24h
USE_REDIS: bool = False