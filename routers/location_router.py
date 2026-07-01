# fleet_api/routers/location_router.py — CORRIGÉ
# Fix : GET /location/all retourne aussi les entrées "INCONNU"
# + endpoint /debug sans auth
# + accept admin ET gestionnaire

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth_router import get_current_user

router = APIRouter(prefix="/location", tags=["GPS"])

class LocationUpdate(BaseModel):
    immatriculation: str
    latitude:        float
    longitude:       float
    vitesse_kmh:     Optional[float] = 0.0
    cap_degrees:     Optional[float] = 0.0
    precision_m:     Optional[float] = 10.0

class LocationResponse(BaseModel):
    immatriculation: str
    latitude:        float
    longitude:       float
    vitesse_kmh:     float
    cap_degrees:     float
    precision_m:     float
    chauffeur_login: str
    updated_at:      str

def _get_db_path() -> str:
    import os
    from pathlib import Path
    # ✅ Chemin calculé dynamiquement — même logique que database.py
    default = str(Path(__file__).parent.parent.parent / "fleet_ai.db")
    return os.environ.get("SQLITE_PATH", default)

def _init_gps_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gps_locations (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            immatriculation  TEXT    NOT NULL,
            latitude         REAL    NOT NULL,
            longitude        REAL    NOT NULL,
            vitesse_kmh      REAL    DEFAULT 0,
            cap_degrees      REAL    DEFAULT 0,
            precision_m      REAL    DEFAULT 10,
            chauffeur_login  TEXT    NOT NULL,
            updated_at       TEXT    NOT NULL,
            UNIQUE(immatriculation)
        )
    """)
    conn.commit()

@router.post("/update")
def update_location(
    loc:  LocationUpdate,
    user: dict = Depends(get_current_user),
):
    import sqlite3
    now = datetime.utcnow().isoformat()
    db  = _get_db_path()
    try:
        conn = sqlite3.connect(db)
        _init_gps_table(conn)
        conn.execute("""
            INSERT INTO gps_locations
                (immatriculation, latitude, longitude, vitesse_kmh,
                 cap_degrees, precision_m, chauffeur_login, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(immatriculation) DO UPDATE SET
                latitude        = excluded.latitude,
                longitude       = excluded.longitude,
                vitesse_kmh     = excluded.vitesse_kmh,
                cap_degrees     = excluded.cap_degrees,
                precision_m     = excluded.precision_m,
                chauffeur_login = excluded.chauffeur_login,
                updated_at      = excluded.updated_at
        """, (
            loc.immatriculation,
            loc.latitude, loc.longitude,
            loc.vitesse_kmh or 0.0,
            loc.cap_degrees or 0.0,
            loc.precision_m or 10.0,
            user.get("login", ""),
            now,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"Erreur GPS SQLite ({db}): {e}")

    return {
        "success": True,
        "immatriculation": loc.immatriculation,
        "updated_at": now,
        "db_path": db,
    }

@router.get("/all", response_model=List[LocationResponse])
def get_all_locations(user: dict = Depends(get_current_user)):
    """Retourne toutes les positions GPS — admin et gestionnaire autorisés."""
    role = user.get("role", "")
    if role not in ("admin", "gestionnaire"):
        raise HTTPException(
            403,
            f"Accès refusé — rôle '{role}' (attendu: admin ou gestionnaire)"
        )
    import sqlite3
    db = _get_db_path()
    try:
        conn = sqlite3.connect(db)
        _init_gps_table(conn)
        rows = conn.execute("""
            SELECT immatriculation, latitude, longitude,
                   vitesse_kmh, cap_degrees, precision_m,
                   chauffeur_login, updated_at
            FROM gps_locations
            ORDER BY updated_at DESC
        """).fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"Erreur lecture GPS ({db}): {e}")

    return [
        LocationResponse(
            immatriculation = r[0],
            latitude        = r[1],
            longitude       = r[2],
            vitesse_kmh     = r[3] or 0.0,
            cap_degrees     = r[4] or 0.0,
            precision_m     = r[5] or 10.0,
            chauffeur_login = r[6] or "",
            updated_at      = r[7] or "",
        )
        for r in rows
    ]

@router.get("/debug")
def debug_locations():
    """Sans auth — pour tester : http://127.0.0.1:8000/location/debug"""
    import sqlite3
    db = _get_db_path()
    try:
        conn = sqlite3.connect(db)
        _init_gps_table(conn)
        rows = conn.execute("""
            SELECT immatriculation, latitude, longitude,
                   chauffeur_login, updated_at
            FROM gps_locations ORDER BY updated_at DESC
        """).fetchall()
        conn.close()
        return {
            "db_path": db,
            "count":   len(rows),
            "positions": [
                {"immat": r[0], "lat": r[1], "lng": r[2],
                 "chauffeur": r[3], "updated_at": r[4]}
                for r in rows
            ]
        }
    except Exception as e:
        return {"error": str(e), "db_path": db}

@router.delete("/clear/{immat}")
def clear_location(immat: str, user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "gestionnaire"):
        raise HTTPException(403, "Accès refusé")
    import sqlite3
    conn = sqlite3.connect(_get_db_path())
    conn.execute("DELETE FROM gps_locations WHERE immatriculation = ?", (immat,))
    conn.commit()
    conn.close()
    return {"deleted": immat}