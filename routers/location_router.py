# fleet_api/routers/location_router.py — CORRIGÉ v3
# Fix principal :
#   UNIQUE(chauffeur_login) au lieu de UNIQUE(immatriculation)
#   → chaque chauffeur garde sa position même s'il envoie INCONNU
#   → plusieurs chauffeurs s'affichent tous sur la carte
#   + GET /location/all enrichit l'immat depuis SQL Server si INCONNU

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
    default = str(Path(__file__).parent.parent.parent / "fleet_ai.db")
    return os.environ.get("SQLITE_PATH", default)

def _init_gps_table(conn):
    """
    UNIQUE sur chauffeur_login (pas immatriculation) :
    → chaque chauffeur a 1 ligne = sa dernière position
    → plusieurs chauffeurs s'affichent tous sur la carte
    """
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
            UNIQUE(chauffeur_login)
        )
    """)
    conn.commit()

def _migrate_table(conn):
    """Migration si l'ancienne table a UNIQUE(immatriculation) → recréer."""
    try:
        # Vérifier le schéma actuel
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='gps_locations'"
        ).fetchone()
        if schema and 'UNIQUE(immatriculation)' in (schema[0] or ''):
            # Sauvegarder les données existantes
            old_rows = conn.execute(
                "SELECT immatriculation, latitude, longitude, vitesse_kmh, "
                "cap_degrees, precision_m, chauffeur_login, updated_at "
                "FROM gps_locations"
            ).fetchall()
            # Supprimer et recréer
            conn.execute("DROP TABLE gps_locations")
            conn.commit()
            _init_gps_table(conn)
            # Réinsérer les données (par chauffeur_login)
            for row in old_rows:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO gps_locations
                        (immatriculation, latitude, longitude, vitesse_kmh,
                         cap_degrees, precision_m, chauffeur_login, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, row)
                except Exception:
                    pass
            conn.commit()
    except Exception:
        pass  # Si migration échoue, continuer normalement

def _get_immat_for_login(login: str) -> Optional[str]:
    """Récupère l'immatriculation assignée à un chauffeur depuis SQL Server."""
    try:
        from routers.fleet import sql_server_engine
        from sqlalchemy import text
        with sql_server_engine.connect() as conn:
            row = conn.execute(text("""
                SELECT TOP 1 immatriculation FROM posseder
                WHERE Utilisateur = :login ORDER BY dd DESC
            """), {"login": login}).mappings().fetchone()
            if row:
                return row["immatriculation"]
    except Exception:
        pass
    return None

# ── POST /location/update ─────────────────────────────────────────────────────
@router.post("/update")
def update_location(
    loc:  LocationUpdate,
    user: dict = Depends(get_current_user),
):
    import sqlite3
    now   = datetime.utcnow().isoformat()
    db    = _get_db_path()
    login = user.get("login", "")

    # ✅ Si immat = INCONNU, essayer de la récupérer depuis SQL Server
    immat = loc.immatriculation
    if not immat or immat == "INCONNU":
        resolved = _get_immat_for_login(login)
        immat = resolved if resolved else f"CHAUFFEUR_{login}"

    try:
        conn = sqlite3.connect(db)
        _migrate_table(conn)   # migration si ancienne table
        _init_gps_table(conn)
        conn.execute("""
            INSERT INTO gps_locations
                (immatriculation, latitude, longitude, vitesse_kmh,
                 cap_degrees, precision_m, chauffeur_login, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chauffeur_login) DO UPDATE SET
                immatriculation = excluded.immatriculation,
                latitude        = excluded.latitude,
                longitude       = excluded.longitude,
                vitesse_kmh     = excluded.vitesse_kmh,
                cap_degrees     = excluded.cap_degrees,
                precision_m     = excluded.precision_m,
                updated_at      = excluded.updated_at
        """, (
            immat,
            loc.latitude, loc.longitude,
            loc.vitesse_kmh or 0.0,
            loc.cap_degrees or 0.0,
            loc.precision_m or 10.0,
            login,
            now,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"Erreur GPS SQLite ({db}): {e}")

    return {
        "success":         True,
        "immatriculation": immat,
        "chauffeur":       login,
        "updated_at":      now,
    }

# ── GET /location/all ─────────────────────────────────────────────────────────
@router.get("/all", response_model=List[LocationResponse])
def get_all_locations(user: dict = Depends(get_current_user)):
    """Retourne toutes les positions GPS — admin et gestionnaire."""
    role = user.get("role", "")
    if role not in ("admin", "gestionnaire"):
        raise HTTPException(
            403, f"Accès refusé — rôle '{role}' (attendu: admin ou gestionnaire)")

    import sqlite3
    db = _get_db_path()
    try:
        conn = sqlite3.connect(db)
        _migrate_table(conn)
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

    result = []
    for r in rows:
        immat = r[0] or ""
        login = r[6] or ""

        # Si l'immat commence par CHAUFFEUR_ ou est vide → résoudre depuis SQL Server
        if not immat or immat.startswith("CHAUFFEUR_") or immat == "INCONNU":
            resolved = _get_immat_for_login(login)
            if resolved:
                immat = resolved

        result.append(LocationResponse(
            immatriculation = immat,
            latitude        = r[1],
            longitude       = r[2],
            vitesse_kmh     = r[3] or 0.0,
            cap_degrees     = r[4] or 0.0,
            precision_m     = r[5] or 10.0,
            chauffeur_login = login,
            updated_at      = r[7] or "",
        ))

    return result

# ── GET /location/debug ───────────────────────────────────────────────────────
@router.get("/debug")
def debug_locations():
    """Sans auth — http://127.0.0.1:8000/location/debug"""
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
            "db_path":   db,
            "count":     len(rows),
            "positions": [
                {"immat": r[0], "lat": r[1], "lng": r[2],
                 "chauffeur": r[3], "updated_at": r[4]}
                for r in rows
            ]
        }
    except Exception as e:
        return {"error": str(e), "db_path": db}

# ── DELETE /location/clear/{immat} ────────────────────────────────────────────
@router.delete("/clear/{immat}")
def clear_location(immat: str, user: dict = Depends(get_current_user)):
    if user.get("role") not in ("admin", "gestionnaire"):
        raise HTTPException(403, "Accès refusé")
    import sqlite3
    conn = sqlite3.connect(_get_db_path())
    conn.execute(
        "DELETE FROM gps_locations WHERE immatriculation = ? OR chauffeur_login = ?",
        (immat, immat))
    conn.commit()
    conn.close()
    return {"deleted": immat}

# ── DELETE /location/clear-all ────────────────────────────────────────────────
@router.delete("/clear-all")
def clear_all_locations(user: dict = Depends(get_current_user)):
    """Vider toute la table GPS — admin uniquement."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin uniquement")
    import sqlite3
    conn = sqlite3.connect(_get_db_path())
    conn.execute("DELETE FROM gps_locations")
    conn.commit()
    conn.close()
    return {"deleted": "all"}