# fleet_api/routers/auth_router.py — CORRIGÉ
# Fix : TYPE "Administrateur" → role "admin" + logging pour debug

from datetime import datetime, timedelta
from typing import Optional
import logging

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import text

logger = logging.getLogger(__name__)

try:
    import jwt
except ImportError:
    raise ImportError("pip install PyJWT")

from routers.fleet import sql_server_engine

router  = APIRouter(prefix="/auth", tags=["Auth"])
bearer  = HTTPBearer(auto_error=False)

JWT_SECRET    = "SmartFleetIA_SECRET_2026_ENSIASD"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_H  = 24

# ── Modèles ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    login:    str
    password: str

class LoginResponse(BaseModel):
    token:   str
    user_id: int
    nom:     str
    prenom:  str
    role:    str
    immat:   Optional[str] = None

class RegisterRequest(BaseModel):
    nom:      str
    prenom:   str
    login:    str
    password: str
    immat:    Optional[str] = None

# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_token(payload: dict) -> str:
    data = payload.copy()
    data["exp"] = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H)
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expiré — reconnectez-vous")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Token invalide")

# ── Mapping TYPE → role ───────────────────────────────────────────────────────

def _map_role(type_raw: str) -> str:
    """
    Mappe le champ TYPE de la table USER vers un rôle normalisé.
    Gère toutes les variantes possibles en BD.
    """
    t = (type_raw or "").strip().lower()
    # Admin / Administrateur / Administrator
    if t in ("admin", "administrateur", "administrator", "administration"):
        return "admin"
    # Chauffeur / Conducteur / Driver
    if t in ("chauffeur", "conducteur", "driver"):
        return "chauffeur"
    # Tout le reste → gestionnaire
    return "gestionnaire"

# ── Dépendances FastAPI ───────────────────────────────────────────────────────

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)
) -> dict:
    if creds is None:
        raise HTTPException(401, "Token manquant")
    return _decode_token(creds.credentials)

def require_gestionnaire(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("gestionnaire", "admin"):
        raise HTTPException(403, "Accès réservé aux gestionnaires")
    return user

def require_chauffeur(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("chauffeur", "admin"):
        raise HTTPException(403, "Accès réservé aux chauffeurs")
    return user

def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "Accès réservé à l'Admin uniquement")
    return user

# ── POST /auth/login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    try:
        with sql_server_engine.connect() as conn:
            row = conn.execute(text("""
                SELECT numero, NOM, PRENOM, LOGIN, PASSEWORD, TYPE
                FROM [USER] WHERE LOGIN = :login
            """), {"login": req.login}).mappings().fetchone()
    except Exception as e:
        raise HTTPException(500, f"Erreur BD : {e}")

    if row is None:
        raise HTTPException(401, "Utilisateur introuvable")
    if row["PASSEWORD"] != req.password:
        raise HTTPException(401, "Mot de passe incorrect")

    # ✅ Mapping robuste — gère "Administrateur", "Chauffeur", etc.
    type_raw = row["TYPE"] or ""
    role     = _map_role(type_raw)

    # Log pour debug
    logger.info(f"Login '{req.login}' → TYPE='{type_raw}' → role='{role}'")

    # Véhicule assigné (chauffeur uniquement)
    immat = None
    if role == "chauffeur":
        try:
            with sql_server_engine.connect() as conn:
                v = conn.execute(text("""
                    SELECT TOP 1 immatriculation FROM posseder
                    WHERE Utilisateur = :login ORDER BY dd DESC
                """), {"login": req.login}).mappings().fetchone()
                if v:
                    immat = v["immatriculation"]
        except Exception:
            pass

    token = _create_token({
        "user_id": int(row["numero"]),
        "login":   req.login,
        "role":    role,
        "immat":   immat,
        "nom":     row["NOM"]    or "",
        "prenom":  row["PRENOM"] or "",
    })

    logger.info(f"Token généré pour '{req.login}' avec role='{role}'")

    return LoginResponse(
        token    = token,
        user_id  = int(row["numero"]),
        nom      = row["NOM"]    or "",
        prenom   = row["PRENOM"] or "",
        role     = role,
        immat    = immat,
    )

# ── POST /auth/register ───────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(req: RegisterRequest, admin: dict = Depends(require_admin)):
    try:
        with sql_server_engine.connect() as conn:
            existing = conn.execute(text("""
                SELECT COUNT(*) as n FROM [USER] WHERE LOGIN = :login
            """), {"login": req.login}).mappings().fetchone()

            if existing and int(existing["n"]) > 0:
                raise HTTPException(400, f"Login '{req.login}' déjà utilisé")

            conn.execute(text("""
                INSERT INTO [USER] (NOM, PRENOM, LOGIN, PASSEWORD, TYPE)
                VALUES (:nom, :prenom, :login, :password, 'Chauffeur')
            """), {
                "nom":      req.nom.upper().strip(),
                "prenom":   req.prenom.strip(),
                "login":    req.login.strip(),
                "password": req.password,
            })

            new_user = conn.execute(text("""
                SELECT numero FROM [USER] WHERE LOGIN = :login
            """), {"login": req.login}).mappings().fetchone()

            if req.immat and new_user:
                try:
                    conn.execute(text("""
                        INSERT INTO posseder
                            (dd, immatriculation, numero_proprietaire, Utilisateur)
                        VALUES (GETDATE(), :immat, 1, :login)
                    """), {"immat": req.immat, "login": req.login})
                except Exception:
                    pass

            conn.commit()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur BD : {e}")

    return {
        "success": True,
        "message": f"Chauffeur {req.prenom} {req.nom.upper()} créé avec succès",
        "login":   req.login,
        "immat":   req.immat,
    }

# ── GET /auth/me ──────────────────────────────────────────────────────────────

@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """Retourne les infos du token courant — utile pour déboguer le rôle."""
    return user

@router.post("/logout")
def logout():
    return {"message": "Déconnecté — supprimez le token localement"}