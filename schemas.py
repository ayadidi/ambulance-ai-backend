"""
schemas.py
==========
Schémas Pydantic v2 — définissent exactement les JSON
retournés par l'API et reçus de Flutter.

Chaque schéma correspond à un endpoint :
  AIVehicleReportSchema → GET /vehicle/{id}/report
  FleetReportSchema     → GET /fleet/report
  VehicleInputSchema    → POST /vehicle/predict
  FeedbackIn            → POST /feedback (corps)
  FeedbackOut           → POST /feedback (réponse)
  ModelsHealthSchema    → GET /models/health
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ── Sous-schémas ──────────────────────────────────────────────────────────────

class TopFeature(BaseModel):
    """Top feature SHAP — affiché dans le graphe Flutter (Écran 2)."""
    name:  str    = Field(description="Nom de la feature (ex: km_depuis_maintenance)")
    value: float  = Field(description="Valeur SHAP signée (>0 augmente le risque)")
    abs:   float  = Field(description="Valeur absolue — pour le tri")


# ── Rapport IA véhicule ───────────────────────────────────────────────────────

class AIVehicleReportSchema(BaseModel):
    """
    Rapport IA complet pour un véhicule.
    Retourné par GET /vehicle/{id}/report et inclus dans /fleet/report.
    """
    immat:        str
    vehicule_id:  int
    timestamp:    float = Field(description="Timestamp UNIX du calcul")

    # ── Scores IA ─────────────────────────────────────────────────────────────
    criticite:           float = Field(ge=0, le=1, description="Score composite 0.5×panne + 0.3×anomalie + 0.2×forecast_ecart")
    priorite:            str   = Field(pattern="^(CRITIQUE|HAUTE|MOYENNE|FAIBLE)$")
    prob_panne:          float = Field(ge=0, le=1, description="XGBoost — probabilité de panne")
    prob_panne_tcn:      float = Field(ge=0, le=1, description="TCN — probabilité de panne")
    score_anomalie:      float = Field(ge=0, le=1, description="max(LSTM-AE, VAE)")
    score_anomalie_ae:   float = Field(ge=0, le=1, description="LSTM Autoencoder normalisé")
    score_anomalie_vae:  float = Field(ge=0, le=1, description="VAE normalisé")
    anomalie_detectee:   bool  = Field(description="True si score_anomalie >= 0.5")

    # ── Prévision carburant N-HiTS ────────────────────────────────────────────
    forecast_carburant:  List[float] = Field(description="Prévision sur 3 mois [L]")
    forecast_ecart_pct:  float       = Field(description="Écart vs moyenne historique (%)")

    # ── Action PPO ────────────────────────────────────────────────────────────
    action_recommandee:  str               = Field(description="Libellé action PPO")
    action_index:        int               = Field(description="Index dans ACTIONS []")
    action_confidence:   float             = Field(description="Confiance PPO ∈ [0,1]")
    action_probs:        Dict[str, float]  = Field(description="Distribution de probabilité sur les 4 actions")

    # ── Explicabilité SHAP ────────────────────────────────────────────────────
    shap_values:         Dict[str, float]  = Field(description="Toutes les SHAP values")
    top_features:        List[TopFeature]  = Field(description="Top 5 features les plus influentes")
    explications:        List[str]         = Field(description="Phrases d'explication en français")

    # ── Méta ──────────────────────────────────────────────────────────────────
    modeles_utilises:    List[str]  = Field(description="Modèles ayant contribué au rapport")
    confidence_globale:  float      = Field(description="Confiance moyenne sur tous les modèles")

    model_config = {"from_attributes": True}


# ── Rapport flotte ────────────────────────────────────────────────────────────

class FleetReportSchema(BaseModel):
    """
    Réponse de GET /fleet/report.
    Contient tous les rapports triés par criticité + statistiques globales.
    """
    total:       int             = Field(description="Nombre total de véhicules")
    critiques:   int             = Field(description="Véhicules en état CRITIQUE")
    hauts:       int             = Field(description="Véhicules en état HAUTE")
    moyens:      int             = Field(description="Véhicules en état MOYENNE")
    faibles:     int             = Field(description="Véhicules en état FAIBLE")
    cache_age_s: Optional[float] = Field(default=None, description="Âge du cache en secondes (None=fraîchement calculé)")
    reports:     List[AIVehicleReportSchema]


# ── Entrée prédiction manuelle ────────────────────────────────────────────────

class VehicleInputSchema(BaseModel):
    """
    Corps de POST /vehicle/predict.
    Toutes les features que l'IA attend — les champs NB-04 non fournis
    sont mis à 0.0 (le modèle a été entraîné avec cette convention).
    """
    immat:                 str   = Field(default="TEST-001", description="Immatriculation")
    vehicule_id:           int   = Field(default=0)

    # Features principales (suffisantes pour un test rapide)
    km_total_6m:           float = Field(default=0.0, ge=0)
    carburant_total_6m:    float = Field(default=0.0, ge=0)
    nb_pleins_6m:          float = Field(default=0.0, ge=0)
    nb_interventions_6m:   float = Field(default=0.0, ge=0)
    cout_intervention_6m:  float = Field(default=0.0, ge=0)
    mois_depuis_debut:     float = Field(default=0.0, ge=0)
    mois_actuel:           int   = Field(default=6, ge=1, le=12)

    # Historique carburant pour N-HiTS (12 valeurs idéalement)
    carburant_historique: Optional[List[float]] = Field(
        default=None,
        description="12 valeurs mensuelles de consommation carburant [L]"
    )

    model_config = {"from_attributes": True}


# ── Feedback technicien ───────────────────────────────────────────────────────

class FeedbackIn(BaseModel):
    """
    Corps de POST /feedback.
    Envoyé par Flutter quand le technicien valide ou rejette une action IA.
    """
    vehicule_id:       int   = Field(description="ID du véhicule concerné")
    immat:             str   = Field(default="", description="Immatriculation (optionnel, pour les logs)")
    action_proposee:   str   = Field(description="Libellé de l'action proposée par PPO")
    decision:          str   = Field(
        pattern="^(approuve|rejete)$",
        description="'approuve' ou 'rejete'"
    )
    commentaire:       str   = Field(default="", description="Commentaire libre du technicien")
    panne_survenue:    Optional[bool] = Field(
        default=None,
        description="Panne réellement survenue après coup ? (pour drift monitoring)"
    )
    invalidate_cache:  bool  = Field(
        default=True,
        description="Invalider le cache de ce véhicule après feedback"
    )


class FeedbackOut(BaseModel):
    """Réponse de POST /feedback."""
    success:     bool
    feedback_id: int    = Field(description="ID de la ligne insérée en SQLite")
    message:     str
    vehicule_id: int
    decision:    str


# ── Santé des modèles ─────────────────────────────────────────────────────────

class ModelsHealthSchema(BaseModel):
    """
    Réponse de GET /models/health.
    Affiché dans l'Écran 5 — Santé modèles de Flutter.
    """
    ready:                bool
    device:               str
    n_features:           int
    models:               Dict[str, bool]           = Field(description="Modèle chargé : True/False")
    meta:                 Dict[str, Any]             = Field(description="Métadonnées du manifest NB-04")
    metriques_reference:  Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    bootstrap_ic95:       List[float]               = Field(default_factory=list)
    feedback_stats:       Dict[str, int]             = Field(default_factory=dict)
    cache_age_s:          Optional[float]            = None
    cache_statut:         str                        = "vide"
    dernier_recalcul:     Optional[Dict[str, Any]]  = None