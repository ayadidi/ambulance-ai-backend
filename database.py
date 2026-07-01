"""
database.py
===========
Couche SQLite — stockage des rapports IA et feedback technicien.

Pourquoi SQLite et pas PostgreSQL ?
  - Zéro configuration : aucun serveur à installer
  - Fichier unique fleet_ai.db dans le dossier projet
  - Performances largement suffisantes pour 27 véhicules
  - Migration vers PostgreSQL triviale : remplacer aiosqlite par
    asyncpg et adapter les placeholders ? → $1 au lieu de ?

Architecture :
  3 tables :
    ai_reports   → cache des rapports IA (TTL 24h)
    ai_feedback  → décisions technicien (POST /feedback)
    fleet_stats  → snapshot global de la flotte par recalcul

Usage dans les routers :
    from database import db
    report = await db.get_cached_report(vehicule_id)
    await db.save_report(vehicule_id, report_dict)
    await db.save_feedback(feedback_dict)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiosqlite

from config import PROJECT_ROOT, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

# Chemin du fichier SQLite (dans le dossier projet, pas dans fleet_api/)
DB_PATH = Path(PROJECT_ROOT) / "fleet_ai.db"

# ─── DDL — création des tables ───────────────────────────────────────────────

_DDL = """
-- Cache des rapports IA par véhicule
CREATE TABLE IF NOT EXISTS ai_reports (
    vehicule_id   INTEGER NOT NULL,
    immat         TEXT    NOT NULL,
    report_json   TEXT    NOT NULL,          -- JSON sérialisé du rapport complet
    criticite     REAL    DEFAULT 0.0,       -- dénormalisé pour ORDER BY rapide
    priorite      TEXT    DEFAULT 'FAIBLE',  -- dénormalisé pour filtres Flutter
    computed_at   REAL    NOT NULL,          -- timestamp UNIX du calcul
    PRIMARY KEY (vehicule_id)
);

-- Feedback technicien (validation / rejet d'une action IA)
CREATE TABLE IF NOT EXISTS ai_feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicule_id    INTEGER NOT NULL,
    immat          TEXT    NOT NULL DEFAULT '',
    action_proposee TEXT   NOT NULL,
    decision       TEXT    NOT NULL CHECK (decision IN ('approuve', 'rejete')),
    commentaire    TEXT    DEFAULT '',
    panne_survenue INTEGER DEFAULT NULL,     -- NULL=inconnu, 0=non, 1=oui
    created_at     REAL    NOT NULL          -- timestamp UNIX
);
CREATE INDEX IF NOT EXISTS idx_feedback_vehicule ON ai_feedback(vehicule_id);
CREATE INDEX IF NOT EXISTS idx_feedback_date ON ai_feedback(created_at DESC);

-- Snapshot global de la flotte (1 ligne par recalcul)
CREATE TABLE IF NOT EXISTS fleet_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    total        INTEGER NOT NULL,
    critiques    INTEGER DEFAULT 0,
    hauts        INTEGER DEFAULT 0,
    moyens       INTEGER DEFAULT 0,
    faibles      INTEGER DEFAULT 0,
    computed_at  REAL    NOT NULL
);
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Classe principale
# ═══════════════════════════════════════════════════════════════════════════════

class FleetDatabase:
    """
    Gestionnaire SQLite asynchrone pour la plateforme Fleet AI.

    Cycle de vie (géré par le lifespan FastAPI) :
        await db.init()    # au démarrage — crée les tables
        await db.close()   # à l'arrêt propre
    """

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Connexion / fermeture ─────────────────────────────────────────────────

    async def init(self) -> None:
        """Ouvre la connexion et crée les tables si elles n'existent pas."""
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row      # accès aux colonnes par nom
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        logger.info(f"  ✅ SQLite initialisé — {self.path}")

    async def close(self) -> None:
        """Ferme proprement la connexion."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("  SQLite — connexion fermée")

    # ── Cache rapports IA ─────────────────────────────────────────────────────

    async def get_cached_report(
        self,
        vehicule_id: int,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> Optional[dict]:
        """
        Retourne le rapport IA mis en cache si son âge < ttl secondes.
        Retourne None si absent ou expiré → le router recalcule.

        Args:
            vehicule_id: identifiant numérique du véhicule
            ttl: durée de validité en secondes (défaut : 24h depuis config)
        """
        if not self._conn:
            return None
        cutoff = time.time() - ttl
        async with self._conn.execute(
            "SELECT report_json, computed_at FROM ai_reports "
            "WHERE vehicule_id = ? AND computed_at > ?",
            (vehicule_id, cutoff),
        ) as cur:
            row = await cur.fetchone()
        if row:
            try:
                return json.loads(row["report_json"])
            except json.JSONDecodeError:
                return None
        return None

    async def save_report(self, vehicule_id: int, report: dict) -> None:
        """
        Sauvegarde (ou remplace) un rapport IA dans le cache.

        Utilise INSERT OR REPLACE pour mettre à jour un rapport existant.
        """
        if not self._conn:
            return
        await self._conn.execute(
            """INSERT OR REPLACE INTO ai_reports
               (vehicule_id, immat, report_json, criticite, priorite, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                vehicule_id,
                report.get("immat", f"AMB-{vehicule_id:03d}"),
                json.dumps(report, ensure_ascii=False),
                float(report.get("criticite", 0.0)),
                str(report.get("priorite", "FAIBLE")),
                time.time(),
            ),
        )
        await self._conn.commit()

    async def save_fleet_reports(self, reports: List[dict]) -> None:
        """Sauvegarde les rapports de toute la flotte en une transaction."""
        if not self._conn:
            return
        now = time.time()
        rows = [
            (
                r.get("vehicule_id", 0),
                r.get("immat", ""),
                json.dumps(r, ensure_ascii=False),
                float(r.get("criticite", 0.0)),
                str(r.get("priorite", "FAIBLE")),
                now,
            )
            for r in reports
        ]
        await self._conn.executemany(
            """INSERT OR REPLACE INTO ai_reports
               (vehicule_id, immat, report_json, criticite, priorite, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )

        # Snapshot global
        critiques = sum(1 for r in reports if r.get("priorite") == "CRITIQUE")
        hauts     = sum(1 for r in reports if r.get("priorite") == "HAUTE")
        moyens    = sum(1 for r in reports if r.get("priorite") == "MOYENNE")
        faibles   = sum(1 for r in reports if r.get("priorite") == "FAIBLE")
        await self._conn.execute(
            "INSERT INTO fleet_stats (total, critiques, hauts, moyens, faibles, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (len(reports), critiques, hauts, moyens, faibles, now),
        )
        await self._conn.commit()
        logger.info(
            f"  Cache SQLite mis à jour : {len(reports)} rapports | "
            f"{critiques} critiques | {hauts} hauts"
        )

    async def get_all_cached_reports(
        self,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> Optional[List[dict]]:
        """
        Retourne TOUS les rapports valides triés par criticité décroissante.
        Retourne None si le cache est vide ou expiré pour au moins 1 véhicule.
        """
        if not self._conn:
            return None
        cutoff = time.time() - ttl
        async with self._conn.execute(
            "SELECT report_json FROM ai_reports "
            "WHERE computed_at > ? "
            "ORDER BY criticite DESC",
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return None
        try:
            return [json.loads(r["report_json"]) for r in rows]
        except json.JSONDecodeError:
            return None

    async def invalidate_cache(self, vehicule_id: Optional[int] = None) -> None:
        """
        Invalide le cache : pour un véhicule (vehicule_id fourni)
        ou pour toute la flotte (vehicule_id=None).
        """
        if not self._conn:
            return
        if vehicule_id is not None:
            await self._conn.execute(
                "DELETE FROM ai_reports WHERE vehicule_id = ?",
                (vehicule_id,),
            )
        else:
            await self._conn.execute("DELETE FROM ai_reports")
        await self._conn.commit()

    # ── Feedback technicien ───────────────────────────────────────────────────

    async def save_feedback(self, feedback: dict) -> int:
        """
        Enregistre la décision du technicien.

        Returns:
            id de la ligne insérée (utile pour confirmer côté Flutter)
        """
        if not self._conn:
            return -1
        panne = feedback.get("panne_survenue")
        if panne is not None:
            panne = int(bool(panne))

        cursor = await self._conn.execute(
            """INSERT INTO ai_feedback
               (vehicule_id, immat, action_proposee, decision,
                commentaire, panne_survenue, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                int(feedback.get("vehicule_id", 0)),
                str(feedback.get("immat", "")),
                str(feedback.get("action_proposee", "")),
                str(feedback.get("decision", "approuve")),
                str(feedback.get("commentaire", "")),
                panne,
                time.time(),
            ),
        )
        await self._conn.commit()
        return cursor.lastrowid or -1

    async def get_feedback_stats(self) -> Dict[str, int]:
        """Retourne des statistiques agrégées sur les feedback (pour drift monitoring)."""
        if not self._conn:
            return {}
        async with self._conn.execute(
            "SELECT decision, COUNT(*) as n FROM ai_feedback GROUP BY decision"
        ) as cur:
            rows = await cur.fetchall()
        return {r["decision"]: r["n"] for r in rows}

    async def get_recent_feedbacks(self, limit: int = 50) -> List[dict]:
        """Retourne les N derniers feedbacks (pour l'écran Planning IA)."""
        if not self._conn:
            return []
        async with self._conn.execute(
            "SELECT * FROM ai_feedback ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Stats flotte ──────────────────────────────────────────────────────────

    async def get_last_fleet_stats(self) -> Optional[dict]:
        """Retourne le dernier snapshot global de la flotte."""
        if not self._conn:
            return None
        async with self._conn.execute(
            "SELECT * FROM fleet_stats ORDER BY computed_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def get_cache_age_seconds(self) -> Optional[float]:
        """Retourne l'âge en secondes du cache le plus récent (None si vide)."""
        if not self._conn:
            return None
        async with self._conn.execute(
            "SELECT MAX(computed_at) as last FROM ai_reports"
        ) as cur:
            row = await cur.fetchone()
        if row and row["last"]:
            return time.time() - row["last"]
        return None


# ── Singleton partagé par toute l'application ────────────────────────────────
db = FleetDatabase()
