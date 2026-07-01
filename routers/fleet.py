"""
routers/fleet.py
================
Endpoints consommés par Flutter — Couche 2 complète.

  GET  /fleet/report              → flotte triée criticité + cache SQLite 24h
  GET  /vehicle/{id}/report       → rapport détaillé 1 véhicule + SHAP
  POST /vehicle/predict           → prédiction manuelle (debug / Swagger)
  POST /feedback                  → technicien valide/rejette une action IA
  GET  /models/health             → état de santé du moteur + métriques

Flux de cache :
  1. Flutter appelle GET /fleet/report
  2. Router vérifie le cache SQLite (TTL 24h)
  3. Cache valide → retour immédiat < 50ms
  4. Cache expiré/vide → recalcul IA (27 véhicules, ~500ms)
                       → sauvegarde SQLite
                       → retour JSON

Pour connecter SQL Server à la place des données simulées :
  Remplacer _load_all_vehicles() et _load_carb_history() par des
  requêtes SQLAlchemy sur ta base existante.
"""

# import logging
# import time
# from typing import List, Optional

# from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
# from fastapi.responses import JSONResponse

# from ai_engine import get_engine
# from database import db
# from schemas import (
#     AIVehicleReportSchema,
#     FeedbackIn,
#     FeedbackOut,
#     FleetReportSchema,
#     ModelsHealthSchema,
#     VehicleInputSchema,
# )

# logger = logging.getLogger(__name__)
# router = APIRouter(tags=["Fleet IA"])


# # ═══════════════════════════════════════════════════════════════════════════════
# # Helpers — données véhicules
# # ═══════════════════════════════════════════════════════════════════════════════

# def _load_all_vehicles() -> List[dict]:
#     """
#     Source de données véhicules.

#     Mode actuel : données simulées reproductibles (seed=42),
#     calibrées sur les stats réelles de la flotte (NB-03).

#     Pour brancher SQL Server :
#     ─────────────────────────
#     from sqlalchemy import create_engine, text

#     _engine = create_engine(
#         "mssql+pyodbc://user:pass@server/bd_ParcAuto"
#         "?driver=ODBC+Driver+17+for+SQL+Server"
#     )
#     with _engine.connect() as conn:
#         rows = conn.execute(text(\"\"\"
#             SELECT
#                 v.id_vehicule            AS vehicule_id,
#                 v.immatriculation        AS immat,
#                 SUM(c.km_parcourus)      AS km_total_6m,
#                 SUM(c.carburant_litre)   AS carburant_total_6m,
#                 COUNT(c.id_carburant)    AS nb_pleins_6m,
#                 COUNT(r.id_reparation)   AS nb_interventions_6m,
#                 SUM(r.cout_total)        AS cout_intervention_6m,
#                 DATEDIFF(MONTH, v.date_mise_service, GETDATE()) AS mois_depuis_debut,
#                 MONTH(GETDATE())         AS mois_actuel
#             FROM Vehicule v
#             LEFT JOIN Carburant  c ON c.id_vehicule = v.id_vehicule
#                                    AND c.date_plein >= DATEADD(MONTH,-6,GETDATE())
#             LEFT JOIN Reparation r ON r.id_vehicule = v.id_vehicule
#                                    AND r.date_reparation >= DATEADD(MONTH,-6,GETDATE())
#             GROUP BY v.id_vehicule, v.immatriculation, v.date_mise_service
#         \"\"\")).fetchall()
#     return [dict(zip(row.keys(), row)) for row in rows]
#     """
#     import random, math
#     random.seed(42)
#     vehicles = []
#     for i in range(1, 28):
#         mois_debut = random.randint(12, 84)
#         n_interv   = random.choices([0, 1, 2, 3, 4, 5], weights=[40, 30, 15, 8, 5, 2])[0]
#         km_base    = random.randint(2000, 8000)
#         cout       = n_interv * random.randint(500, 8000) if n_interv > 0 else 0
#         vehicles.append({
#             "immat":               f"AMB-{i:03d}",
#             "vehicule_id":         i,
#             "km_total_6m":         km_base,
#             "carburant_total_6m":  round(km_base * 0.08, 1),
#             "nb_pleins_6m":        max(1, km_base // 500),
#             "nb_interventions_6m": n_interv,
#             "cout_intervention_6m": cout,
#             "mois_depuis_debut":   mois_debut,
#             "mois_actuel":         6,
#         })
#     return vehicles


# def _load_vehicle(vehicule_id: int) -> dict:
#     """Retourne un véhicule par ID (404 si introuvable)."""
#     for v in _load_all_vehicles():
#         if v["vehicule_id"] == vehicule_id:
#             return v
#     raise HTTPException(
#         status_code=404,
#         detail=f"Véhicule {vehicule_id} introuvable dans la flotte"
#     )


# def _load_carb_history(vehicule_id: int) -> List[float]:
#     """
#     Historique carburant mensuel sur 12 mois.

#     Pour brancher SQL Server :
#     ─────────────────────────
#     with _engine.connect() as conn:
#         rows = conn.execute(text(\"\"\"
#             SELECT MONTH(date_plein) AS mois, SUM(carburant_litre) AS total
#             FROM Carburant
#             WHERE id_vehicule = :vid
#               AND date_plein >= DATEADD(MONTH, -12, GETDATE())
#             GROUP BY MONTH(date_plein)
#             ORDER BY mois
#         \"\"\"), {"vid": vehicule_id}).fetchall()
#     return [r.total for r in rows]
#     """
#     import random, math
#     random.seed(vehicule_id * 7)
#     base = random.uniform(100, 500)
#     return [
#         round(base * (1 + 0.1 * math.sin(2 * math.pi * m / 12)
#                       + random.gauss(0, 0.05)), 1)
#         for m in range(12)
#     ]





# Contenu corrigé et connecté pour routers/fleet.py
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from sqlalchemy import create_engine, text

from ai_engine import get_engine
from database import db
from config import SQL_SERVER_URL # Importation de la chaîne Windows Auth
from schemas import (
    AIVehicleReportSchema,
    FeedbackIn,
    FeedbackOut,
    FleetReportSchema,
    ModelsHealthSchema,
    VehicleInputSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Fleet IA"])

# Initialisation du moteur SQLAlchemy pour SQL Server
sql_server_engine = create_engine(
    SQL_SERVER_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={"timeout": 60},   # timeout connexion SQL Server (secondes)
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — Connexion Réelle SQL Server (bd_ParcAuto)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_vehicles() -> List[dict]:
    """
    Source de données réelle — Extrait les données de la base SQL Server bd_ParcAuto.

    Mapping colonnes réelles (NB-03) :
      Ambulance.id_vehicule        → vehicule_id
      Vehicule.immatriculation     → immat  (clé de jointure entre toutes les tables)
      Vehicule.dmc                 → date_mise_service
      Carburant.dateCarb           → date du plein (ex date_plein)
      Carburant.quantite           → litres carburant (ex carburant_litre)
      Carburant.numero             → id du plein (ex id_carburant)
      Kilometrage.(compteurFin-compteurDebut) → km parcourus par mois
      reparation.dateCarb          → date réparation (ex date_reparation)
      reparation.numero            → id réparation
      DetailsReparation.total      → coût réparation (ex cout_total)
    """
    try:
        with sql_server_engine.connect() as conn:
            query = text("""
                WITH VehiculeBase AS (
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY immatriculation) AS vehicule_id,
                        immatriculation,
                        dmc
                    FROM Vehicule
                ),
                CarburantAgg AS (
                    SELECT immatriculation,
                           SUM(quantite)       AS carburant_total_6m,
                           COUNT(numero)       AS nb_pleins_6m
                    FROM Carburant
                    WHERE dateCarb >= DATEADD(MONTH, -6, GETDATE())
                    GROUP BY immatriculation
                ),
                KmAgg AS (
                    SELECT immatriculation,
                           SUM(CAST(compteurFin AS BIGINT) - CAST(compteurDebut AS BIGINT)) AS km_total_6m
                    FROM Kilometrage
                    WHERE DATEFROMPARTS(annee, mois, 1) >= DATEADD(MONTH, -6, GETDATE())
                      AND compteurFin > compteurDebut
                    GROUP BY immatriculation
                ),
                ReparationAgg AS (
                    SELECT r.immatriculation,
                           COUNT(DISTINCT r.numero)  AS nb_interventions_6m,
                           ISNULL(SUM(dr.total), 0)  AS cout_intervention_6m
                    FROM reparation r
                    LEFT JOIN DetailsReparation dr ON dr.codeBon = r.NSerie
                    WHERE r.dateCarb >= DATEADD(MONTH, -6, GETDATE())
                    GROUP BY r.immatriculation
                )
                SELECT
                    vb.vehicule_id,
                    vb.immatriculation                              AS immat,
                    ISNULL(km.km_total_6m, 0)                      AS km_total_6m,
                    ISNULL(ca.carburant_total_6m, 0)               AS carburant_total_6m,
                    ISNULL(ca.nb_pleins_6m, 0)                     AS nb_pleins_6m,
                    ISNULL(ra.nb_interventions_6m, 0)              AS nb_interventions_6m,
                    ISNULL(ra.cout_intervention_6m, 0)             AS cout_intervention_6m,
                    ISNULL(DATEDIFF(MONTH, vb.dmc, GETDATE()), 0)  AS mois_depuis_debut,
                    MONTH(GETDATE())                               AS mois_actuel
                FROM VehiculeBase vb
                LEFT JOIN CarburantAgg ca ON ca.immatriculation = vb.immatriculation
                LEFT JOIN KmAgg        km ON km.immatriculation = vb.immatriculation
                LEFT JOIN ReparationAgg ra ON ra.immatriculation = vb.immatriculation
            """)
            result = conn.execute(query)
            # Conversion propre des lignes SQL Server en liste de dictionnaires Python
            return [dict(row) for row in result.mappings()]
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction SQL Server (_load_all_vehicles) : {e}")
        raise HTTPException(status_code=500, detail="Erreur de liaison bd_ParcAuto")


def _load_vehicle(vehicule_id: int) -> dict:
    """Retourne un véhicule spécifique depuis SQL Server."""
    for v in _load_all_vehicles():
        if v["vehicule_id"] == vehicule_id:
            return v
    raise HTTPException(
        status_code=404,
        detail=f"Véhicule {vehicule_id} introuvable dans la flotte bd_ParcAuto"
    )


def _load_carb_history(vehicule_id: int) -> List[float]:
    """
    Historique carburant mensuel sur 12 mois extrait depuis SQL Server pour N-HiTS.
    """
    try:
        with sql_server_engine.connect() as conn:
            # Historique 12 mois via immatriculation (résolu depuis vehicule_id)
            query = text("""
                SELECT MONTH(c.dateCarb) AS mois, ISNULL(SUM(c.quantite), 0) AS total
                FROM Carburant c
                INNER JOIN (
                    SELECT immatriculation,
                           ROW_NUMBER() OVER (ORDER BY immatriculation) AS rn
                    FROM Vehicule
                ) vb ON vb.immatriculation = c.immatriculation AND vb.rn = :vid
                WHERE c.dateCarb >= DATEADD(MONTH, -12, GETDATE())
                GROUP BY MONTH(c.dateCarb)
                ORDER BY mois
            """)
            result = conn.execute(query, {"vid": vehicule_id})
            history = [float(row["total"]) for row in result.mappings()]
            
            # Sécurité si l'historique est trop court (remplissage par des 0.0 pour N-HiTS)
            if len(history) < 12:
                history = [0.0] * (12 - len(history)) + history
            return history
    except Exception as e:
        logger.error(f"Erreur historique carburant SQL Server pour ID {vehicule_id} : {e}")
        return [0.0] * 12


def _load_all_carb_histories(vehicles: List[dict]) -> dict:
    """
    Charge l'historique carburant 12 mois pour TOUS les véhicules en une seule requête.
    Retourne un dict { vehicule_id: [float x12] }
    """
    if not vehicles:
        return {}

    # Map immat -> vehicule_id
    immat_to_id = {v["immat"]: v["vehicule_id"] for v in vehicles}
    immat_list  = list(immat_to_id.keys())

    try:
        with sql_server_engine.connect() as conn:
            placeholders = ", ".join(f":im{i}" for i in range(len(immat_list)))
            params = {f"im{i}": immat for i, immat in enumerate(immat_list)}
            query = text(f"""
                SELECT c.immatriculation,
                       MONTH(c.dateCarb) AS mois,
                       ISNULL(SUM(c.quantite), 0) AS total
                FROM Carburant c
                WHERE c.immatriculation IN ({placeholders})
                  AND c.dateCarb >= DATEADD(MONTH, -12, GETDATE())
                GROUP BY c.immatriculation, MONTH(c.dateCarb)
            """)
            rows = list(conn.execute(query, params).mappings())
    except Exception as e:
        logger.error(f"Erreur _load_all_carb_histories : {{e}}")
        rows = []

    # Construire le dict par vehicule_id
    from collections import defaultdict
    raw: dict = defaultdict(dict)
    for row in rows:
        vid = immat_to_id.get(row["immatriculation"])
        if vid is not None:
            raw[vid][int(row["mois"])] = float(row["total"])

    result = {}
    for v in vehicles:
        vid = v["vehicule_id"]
        monthly = raw.get(vid, {})
        history = [monthly.get(m, 0.0) for m in range(1, 13)]
        result[vid] = history
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# Le reste du fichier (endpoints, tâches de fond, feedback) reste identique...
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Tâche de fond — recalcul et mise en cache
# ═══════════════════════════════════════════════════════════════════════════════

async def _recalculate_and_cache() -> List[dict]:
    """
    Recalcule les rapports IA pour toute la flotte et met à jour le cache.
    Appelé quand le cache est vide ou expiré.
    """
    t0 = time.perf_counter()
    engine   = get_engine()
    vehicles = _load_all_vehicles()
    # Une seule requête SQL pour tout l'historique (évite 27 appels séparés)
    carb_histories = _load_all_carb_histories(vehicles)

    reports      = engine.predict_fleet(vehicles, carb_histories)
    reports_dicts = [r.to_dict() for r in reports]

    # Sauvegarde asynchrone en SQLite
    await db.save_fleet_reports(reports_dicts)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"  Recalcul flotte : {len(reports)} véhicules en {elapsed:.0f}ms")
    return reports_dicts


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GET /fleet/report
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/fleet/report",
    response_model=FleetReportSchema,
    summary="Rapport IA complet de la flotte",
    description=(
        "Retourne les 27 rapports IA triés par criticité décroissante. "
        "Cache SQLite 24h — réponse < 50ms si cache valide, ~500ms sinon. "
        "Appelé par l'écran IA Home de Flutter."
    ),
)
async def get_fleet_report(
    force_refresh: bool = Query(
        default=False,
        description="Forcer le recalcul même si le cache est valide"
    ),
):
    try:
        t0 = time.perf_counter()

        # ── Étape 1 : Vérifier le cache SQLite ───────────────────────────────
        reports_dicts: Optional[List[dict]] = None

        if not force_refresh:
            reports_dicts = await db.get_all_cached_reports()
            if reports_dicts:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"Fleet report (cache) : {len(reports_dicts)} véhicules | "
                    f"{elapsed:.0f}ms"
                )

        # ── Étape 2 : Recalcul si cache vide/expiré ──────────────────────────
        if reports_dicts is None:
            reports_dicts = await _recalculate_and_cache()

        # ── Étape 3 : Statistiques globales ──────────────────────────────────
        critiques = sum(1 for r in reports_dicts if r.get("priorite") == "CRITIQUE")
        hauts     = sum(1 for r in reports_dicts if r.get("priorite") == "HAUTE")
        moyens    = sum(1 for r in reports_dicts if r.get("priorite") == "MOYENNE")
        faibles   = sum(1 for r in reports_dicts if r.get("priorite") == "FAIBLE")

        age = await db.get_cache_age_seconds()

        return {
            "total":       len(reports_dicts),
            "critiques":   critiques,
            "hauts":       hauts,
            "moyens":      moyens,
            "faibles":     faibles,
            "cache_age_s": round(age, 0) if age is not None else None,
            "reports":     reports_dicts,
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Moteur IA non prêt : {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur get_fleet_report")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GET /vehicle/{vehicule_id}/report
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/vehicle/{vehicule_id}/report",
    response_model=AIVehicleReportSchema,
    summary="Rapport IA détaillé d'un véhicule",
    description=(
        "Retourne le rapport complet incluant les SHAP values, "
        "la courbe de prévision carburant N-HiTS, et l'action PPO. "
        "Appelé quand l'utilisateur tape sur une carte dans Flutter."
    ),
)
async def get_vehicle_report(
    vehicule_id: int,
    force_refresh: bool = Query(
        default=False,
        description="Ignorer le cache et recalculer"
    ),
):
    try:
        t0 = time.perf_counter()

        # ── Cache individuel ──────────────────────────────────────────────────
        if not force_refresh:
            cached = await db.get_cached_report(vehicule_id)
            if cached:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"Vehicle {vehicule_id} report (cache) : {elapsed:.0f}ms"
                )
                return cached

        # ── Recalcul ──────────────────────────────────────────────────────────
        engine  = get_engine()
        vehicle = _load_vehicle(vehicule_id)  # lève 404 si introuvable
        history = _load_carb_history(vehicule_id)
        report  = engine.predict_vehicle(vehicle, carb_history=history)
        d       = report.to_dict()

        # Sauvegarder dans le cache
        await db.save_report(vehicule_id, d)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Vehicle {vehicule_id} report (fresh) : "
            f"criticité={d['criticite']:.3f} | {elapsed:.0f}ms"
        )
        return d

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Moteur IA non prêt : {e}")
    except Exception as e:
        logger.exception(f"Erreur get_vehicle_report({vehicule_id})")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. POST /vehicle/predict — prédiction manuelle
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/vehicle/predict",
    response_model=AIVehicleReportSchema,
    summary="Prédiction manuelle (debug / Swagger UI)",
    description=(
        "Calcule un rapport IA à partir de données saisies manuellement. "
        "Utile pour tester depuis Swagger ou depuis un formulaire Flutter."
    ),
)
async def predict_vehicle_manual(data: VehicleInputSchema):
    try:
        engine  = get_engine()
        vehicle = data.model_dump(exclude={"carburant_historique"})
        history = data.carburant_historique
        report  = engine.predict_vehicle(vehicle, carb_history=history)
        return report.to_dict()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Moteur IA non prêt : {e}")
    except Exception as e:
        logger.exception("Erreur predict_vehicle_manual")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. POST /feedback — validation technicien
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/feedback",
    response_model=FeedbackOut,
    summary="Validation technicien d'une action IA",
    description=(
        "Enregistre la décision du technicien (approuver/rejeter l'action recommandée). "
        "Stocké en SQLite. Utilisé pour le monitoring de drift et le re-training. "
        "Si la panne est confirmée (panne_survenue=true), le score de drift augmente."
    ),
)
async def submit_feedback(
    feedback: FeedbackIn,
    background_tasks: BackgroundTasks,
):
    try:
        feedback_dict = feedback.model_dump()

        # ── Invalider le cache si feedback → forcer recalcul au prochain appel
        if feedback.invalidate_cache:
            background_tasks.add_task(
                db.invalidate_cache, feedback.vehicule_id
            )

        row_id = await db.save_feedback(feedback_dict)

        logger.info(
            f"Feedback #{row_id} : AMB-{feedback.vehicule_id:03d} | "
            f"{feedback.action_proposee} → {feedback.decision.upper()} | "
            f"panne_réelle={feedback.panne_survenue}"
        )

        return {
            "success":    True,
            "feedback_id": row_id,
            "message":    (
                f"Feedback enregistré pour AMB-{feedback.vehicule_id:03d}. "
                f"Action '{feedback.action_proposee}' : {feedback.decision.upper()}."
            ),
            "vehicule_id": feedback.vehicule_id,
            "decision":    feedback.decision,
        }
    except Exception as e:
        logger.exception("Erreur submit_feedback")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GET /models/health
# ═══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/models/health",
    response_model=ModelsHealthSchema,
    summary="État de santé du moteur IA",
    description=(
        "Retourne le statut de chaque modèle, les métriques de référence "
        "(AUC, MAPE, Recall depuis NB-12/NB-13), et les statistiques de feedback. "
        "Affiché dans l'écran 'Santé modèles' de Flutter."
    ),
)
async def get_models_health():
    try:
        engine = get_engine()
        health = engine.health()

        # Métriques de référence depuis NB-12 / NB-13
        health["metriques_reference"] = {
            "xgboost": {"auc": 0.818, "recall": 0.951, "f1": 0.436, "type": "classification"},
            "tcn":     {"auc": 0.852, "recall": 0.904, "f1": 0.202, "type": "classification"},
            "lstm_ae": {"auc": 0.820, "recall": 0.976, "f1": 0.383, "type": "anomalie"},
            "vae":     {"auc": 0.800, "recall": 0.960, "f1": 0.350, "type": "anomalie"},
            "nhits":   {"mape": 8.42,  "r2": 0.884, "type": "forecast"},
            "tft":     {"mape": 9.15,  "r2": 0.821, "type": "forecast"},
            "ppo":     {"reward_moyen": 0.87, "disponibilite": 1.0, "type": "rl"},
        }
        health["bootstrap_ic95"] = [0.766, 0.869]

        # Stats feedback depuis SQLite
        health["feedback_stats"] = await db.get_feedback_stats()

        # Âge du cache
        age = await db.get_cache_age_seconds()
        health["cache_age_s"]    = round(age, 0) if age is not None else None
        health["cache_statut"]   = (
            "valide"  if age is not None and age < 86400 else
            "expiré"  if age is not None else
            "vide"
        )

        # Dernier snapshot flotte
        last_stats = await db.get_last_fleet_stats()
        health["dernier_recalcul"] = last_stats

        return health

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Moteur IA non prêt : {e}")
    except Exception as e:
        logger.exception("Erreur get_models_health")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DELETE /cache — invalidation manuelle (admin)
# ═══════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/cache",
    summary="Invalider le cache (admin)",
    description="Force le recalcul au prochain appel GET /fleet/report.",
    tags=["Admin"],
)
async def clear_cache(vehicule_id: Optional[int] = Query(
    default=None,
    description="ID d'un véhicule spécifique (None = toute la flotte)"
)):
    await db.invalidate_cache(vehicule_id)
    target = f"AMB-{vehicule_id:03d}" if vehicule_id else "toute la flotte"
    return {"success": True, "message": f"Cache invalidé pour {target}"}