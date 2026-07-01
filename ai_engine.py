"""
ai_engine.py
=============
Noyau IA de la plateforme Fleet AI — chargement des 5 modèles en mémoire,
pipeline de normalisation (RobustScaler), calcul du rapport IA par véhicule
(AIVehicleReport), et score de criticité composite.

Architectures extraites des notebooks réels :
  - TCNClassifier / TransformerClassifier  → NB-06 V3
  - LSTMAutoencoder / VAE                  → NB-07 V3
  - NHitsV2 / TFTSimplified               → NB-08 V2
  - ActorCritic (PPO)                      → NB-09

Usage :
    engine = AIEngine(project_root="C:/Users/.../projet")
    engine.load()
    report = engine.predict_vehicle(vehicle_data)
    fleet   = engine.predict_fleet(all_vehicles_data)
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ARCHITECTURES PYTORCH (copiées à l'identique des notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

# ── NB-06 V3 : TCN ────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        out = self.act(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(self.act(self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2)))
        return out + res


class TCNClassifier(nn.Module):
    def __init__(self, n_features=44, n_channels=32, n_layers=3, kernel_size=3, dropout=0.3):
        super().__init__()
        channels = [n_features] + [n_channels] * n_layers
        self.tcn_blocks = nn.ModuleList([
            TCNResBlock(channels[i], channels[i+1], kernel_size, 2**i, dropout)
            for i in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.bn   = nn.BatchNorm1d(n_channels)
        self.head = nn.Linear(n_channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)
        x = self.bn(x)
        x = self.drop(x)
        return self.head(x).squeeze(-1)


# ── NB-06 V3 : Transformer ────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class PreLNTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        xn = self.norm1(x)
        x  = x + self.attn(xn, xn, xn, attn_mask=mask, need_weights=False)[0]
        x  = x + self.ff(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, n_features=44, d_model=48, n_heads=4, n_layers=2, d_ff=192, dropout=0.25):
        super().__init__()
        self.proj     = nn.Linear(n_features, d_model)
        self.pos_enc  = PositionalEncoding(d_model, dropout=dropout)
        self.norm_in  = nn.LayerNorm(d_model)
        self.layers   = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.pool_w   = nn.Linear(d_model, 1)
        self.drop     = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.norm_in(self.proj(x))
        x = self.pos_enc(x)
        W    = x.size(1)
        mask = torch.triu(torch.ones(W, W, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm_out(x)
        w = torch.softmax(self.pool_w(x), dim=1)
        x = (w * x).sum(dim=1)
        return self.head(self.drop(x)).squeeze(-1)


# ── NB-07 V3 : LSTM Autoencoder ───────────────────────────────────────────────
class LSTMEncoder(nn.Module):
    def __init__(self, n_features=44, hidden_dim=64, latent_dim=32, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.act(self.fc(h_n[-1]))


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, n_features=44, n_layers=2, seq_len=1, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_dim, n_features)

    def forward(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=44, hidden_dim=64, latent_dim=32, n_layers=2, seq_len=1, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, n_features, n_layers, seq_len, dropout)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            err   = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy()


# ── NB-07 V3 : VAE ────────────────────────────────────────────────────────────
class VAEEncoder(nn.Module):
    def __init__(self, n_features=44, hidden_dims=(128, 64), latent_dim=16):
        super().__init__()
        layers, in_d = [], n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
            in_d = h
        self.net    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_d, latent_dim)
        self.fc_logv= nn.Linear(in_d, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logv(h).clamp(-4, 4)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dims=(128, 64), n_features=44):
        super().__init__()
        layers, in_d = [], latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
            in_d = h
        layers.append(nn.Linear(in_d, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, n_features=44, hidden_dims=(128, 64), latent_dim=16):
        super().__init__()
        self.encoder = VAEEncoder(n_features, hidden_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dims, n_features)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def anomaly_score(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_hat, mu, log_var = self.forward(x)
            recon = ((x - x_hat) ** 2).mean(dim=1)
            kl    = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)
        return (recon + 0.1 * kl).cpu().numpy()


# ── NB-08 V2 : N-HiTS ────────────────────────────────────────────────────────
class NHitsBlock(nn.Module):
    def __init__(self, window, horizon, pool_size, n_hidden=32, dropout=0.2):
        super().__init__()
        pool_win = (window + pool_size - 1) // pool_size
        self.pool     = nn.AvgPool1d(pool_size, stride=pool_size, padding=0)
        self.fc1      = nn.Linear(pool_win, n_hidden)
        self.fc2      = nn.Linear(n_hidden, n_hidden)
        self.backcast = nn.Linear(n_hidden, window)
        self.forecast = nn.Linear(n_hidden, horizon)
        self.drop     = nn.Dropout(dropout)
        self.act      = nn.GELU()
        self.norm     = nn.LayerNorm(n_hidden)
        self._in_size = pool_win

    def forward(self, x):
        xp = self.pool(x.unsqueeze(1)).squeeze(1)
        if xp.size(1) < self._in_size:
            xp = F.pad(xp, (0, self._in_size - xp.size(1)))
        elif xp.size(1) > self._in_size:
            xp = xp[:, :self._in_size]
        h  = self.act(self.norm(self.fc1(xp)))
        h  = self.drop(self.act(self.fc2(h)))
        return self.backcast(h), self.forecast(h).squeeze(-1)


class NHitsV2(nn.Module):
    def __init__(self, window=12, horizon=1, n_hidden=32, dropout=0.2):
        super().__init__()
        pool_sizes  = [max(1, window // 3), max(1, window // 6), 1]
        self.blocks = nn.ModuleList([
            NHitsBlock(window, horizon, ps, n_hidden, dropout) for ps in pool_sizes
        ])
        self.W = window

    def forward(self, x):
        residual   = x.clone()
        total_fc   = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            bc, fc = block(residual)
            if bc.size(1) != self.W:
                bc = F.interpolate(bc.unsqueeze(1), size=self.W,
                                   mode="linear", align_corners=False).squeeze(1)
            residual  = residual - bc
            total_fc  = total_fc + fc
        return total_fc


# ── NB-08 V2 : TFT ───────────────────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ELU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TFTSimplified(nn.Module):
    def __init__(self, window=12, horizon=1, d_model=32, n_heads=2, dropout=0.15):
        super().__init__()
        self.W          = window
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = nn.Embedding(window, d_model)
        self.grn_enc    = GatedResidualNetwork(d_model, d_model * 2, dropout)
        self.attn       = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn  = nn.LayerNorm(d_model)
        self.grn_dec    = GatedResidualNetwork(d_model, d_model * 2, dropout)
        self.pool       = nn.Linear(d_model, 1)
        self.head       = nn.Linear(window, horizon)

    def forward(self, x):
        h   = self.input_proj(x.unsqueeze(-1))
        pos = torch.arange(self.W, device=x.device)
        h   = h + self.pos_enc(pos).unsqueeze(0)
        h   = self.grn_enc(h)
        mask = torch.triu(torch.ones(self.W, self.W, device=x.device), 1).bool()
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        h   = self.norm_attn(h + attn_out)
        h   = self.grn_dec(h)
        h   = self.pool(h).squeeze(-1)
        return self.head(h).squeeze(-1)


# ── NB-09 : PPO ActorCritic ──────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=216, n_actions=4, hidden=(256, 128)):
        super().__init__()
        layers, in_d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.shared = nn.Sequential(*layers)
        self.actor  = nn.Linear(in_d, n_actions)
        self.critic = nn.Linear(in_d, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action_probs(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs     = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATACLASS — RAPPORT IA PAR VÉHICULE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AIVehicleReport:
    """Rapport IA complet pour un véhicule — sortie unifiée du moteur IA."""
    immat:               str
    vehicule_id:         int
    timestamp:           float = field(default_factory=time.time)

    # Scores IA (tous dans [0, 1])
    prob_panne:          float = 0.0   # XGBoost ou TCN
    prob_panne_tcn:      float = 0.0   # TCN
    score_anomalie_ae:   float = 0.0   # LSTM-AE (normalisé)
    score_anomalie_vae:  float = 0.0   # VAE (normalisé)
    score_anomalie:      float = 0.0   # max(ae, vae)

    # Prévision carburant
    forecast_carburant:  List[float] = field(default_factory=list)   # 3 mois
    forecast_horizon:    int = 3
    forecast_ecart_pct:  float = 0.0   # écart vs moyenne historique

    # PPO
    action_recommandee:  str = "Idle / Maintien en service"
    action_index:        int = 0
    action_confidence:   float = 0.0
    action_probs:        Dict[str, float] = field(default_factory=dict)

    # Criticité composite
    criticite:           float = 0.0   # 0.5×panne + 0.3×anom + 0.2×fc_ecart
    priorite:            str = "FAIBLE"  # CRITIQUE / HAUTE / MOYENNE / FAIBLE

    # Explicabilité SHAP
    shap_values:         Dict[str, float] = field(default_factory=dict)
    top_features:        List[Dict] = field(default_factory=list)
    explications:        List[str] = field(default_factory=list)

    # Meta
    anomalie_detectee:   bool = False
    modeles_utilises:    List[str] = field(default_factory=list)
    confidence_globale:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "immat":             self.immat,
            "vehicule_id":       self.vehicule_id,
            "timestamp":         self.timestamp,
            "criticite":         round(self.criticite, 4),
            "priorite":          self.priorite,
            "prob_panne":        round(self.prob_panne, 4),
            "prob_panne_tcn":    round(self.prob_panne_tcn, 4),
            "score_anomalie":    round(self.score_anomalie, 4),
            "score_anomalie_ae": round(self.score_anomalie_ae, 4),
            "score_anomalie_vae":round(self.score_anomalie_vae, 4),
            "anomalie_detectee": self.anomalie_detectee,
            "forecast_carburant":self.forecast_carburant,
            "forecast_ecart_pct":round(self.forecast_ecart_pct, 2),
            "action_recommandee":self.action_recommandee,
            "action_index":      self.action_index,
            "action_confidence": round(self.action_confidence, 4),
            "action_probs":      {k: round(v, 4) for k, v in self.action_probs.items()},
            "shap_values":       {k: round(v, 4) for k, v in self.shap_values.items()},
            "top_features":      self.top_features,
            "explications":      self.explications,
            "modeles_utilises":  self.modeles_utilises,
            "confidence_globale":round(self.confidence_globale, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MOTEUR IA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

ACTIONS = [
    "Idle / Maintien en service",
    "Inspection technique",
    "Maintenance préventive",
    "Maintenance corrective",
]


def _find_file(root: Path, *candidates: str) -> Optional[Path]:
    """Cherche un fichier parmi plusieurs chemins relatifs à la racine projet."""
    for candidate in candidates:
        p = root / candidate
        if p.exists():
            return p
    return None


class AIEngine:
    """
    Moteur IA central — charge les 5 modèles une seule fois au démarrage
    du serveur FastAPI, puis produit un AIVehicleReport par véhicule.

    Paramètres
    ----------
    project_root : str | Path
        Racine du projet (dossier contenant outputs/, fleet_ai_simple/, etc.)
        Exemple Windows : "C:/Users/diaya/les cours/ensiasd/stage/projet"
    device : str
        "cpu" ou "cuda" (cpu recommandé pour le déploiement initial)
    """

    def __init__(self, project_root: str | Path, device: str = "cpu"):
        self.root   = Path(project_root)
        self.device = torch.device(device)
        self.ready  = False

        # Modèles chargés
        self._scaler       = None
        self._xgb          = None
        self._tcn          = None
        self._transformer  = None
        self._lstm_ae      = None
        self._vae          = None
        self._nhits        = None
        self._tft          = None
        self._ppo          = None

        # Métadonnées des checkpoints
        self._meta: Dict = {}

        # SHAP explainer (initialisé après chargement XGBoost)
        self._xgb_booster    = None   # Booster re-sauvegardé JSON (sans mismatch pkl)
        self._shap_explainer = None
        self._shap_mode      = "tree"  # "tree" | "kernel"

        # Noms de features (chargés depuis manifest NB-04)
        self._feature_cols: List[str] = []
        self._n_features: int = 44

        # Seuils anomalie
        self._thr_ae:  float = 0.25
        self._thr_vae: float = 48.0

        # Normalisation forecasting
        self._nhits_mu:    float = 0.0
        self._nhits_sigma: float = 1.0
        self._tft_mu:      float = 0.0
        self._tft_sigma:   float = 1.0

        logger.info(f"AIEngine initialisé | root={self.root} | device={self.device}")

    # ─── Chargement ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Charge tous les modèles. Appeler une seule fois au démarrage."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  Chargement des modèles — AI Engine")
        logger.info("=" * 60)

        self._load_manifest()
        self._load_scaler()
        self._load_xgboost()
        self._load_tcn()
        self._load_transformer()
        self._load_lstm_ae()
        self._load_vae()
        self._load_nhits()
        self._load_tft()
        self._load_ppo()
        self._init_shap()

        n_ok = sum(1 for m in [
            self._scaler, self._xgb, self._tcn, self._transformer,
            self._lstm_ae, self._vae, self._nhits, self._tft, self._ppo
        ] if m is not None)

        logger.info("=" * 60)
        logger.info(f"  {n_ok}/9 modèles chargés en {time.time()-t0:.1f}s")
        logger.info("=" * 60)
        self.ready = True

    def _load_manifest(self):
        """Charge manifest NB-04 pour récupérer la liste exacte des features."""
        for candidate in [
            "outputs/NB04_V4/reports/manifest_nb04_v4.json",
            "outputs/NB04_V3/reports/manifest_nb04_v3.json",
            "outputs/NB04/reports/manifest_nb04.json",
        ]:
            p = self.root / candidate
            if p.exists():
                with open(p) as f:
                    manifest = json.load(f)
                self._feature_cols = manifest.get("feature_cols", [])
                self._n_features   = manifest.get("n_features", 44)
                logger.info(f"  ✅ Manifest NB-04 : {self._n_features} features")
                return
        logger.warning("  ⚠️  Manifest NB-04 introuvable — n_features=44 par défaut")
        self._n_features = 44

    def _load_scaler(self):
        p = _find_file(self.root,
            "outputs/NB04_V4/data/robust_scaler_v4.pkl",
            "outputs/NB04_V3/data/robust_scaler_v3.pkl",
            "outputs/NB04/data/robust_scaler.pkl",
        )
        if p:
            try:
                self._scaler = joblib.load(p)
                logger.info(f"  ✅ RobustScaler ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ RobustScaler : {e}")
        else:
            logger.warning("  ❌ RobustScaler introuvable")

    def _load_xgboost(self):
        p = _find_file(self.root,
            "outputs/NB13/reports/xgb_production_model.pkl",
            "outputs/NB05/reports/xgb_model.pkl",
        )
        if p:
            try:
                import pickle as pkl
                with open(p, "rb") as f:
                    self._xgb = pkl.load(f)
                logger.info(f"  ✅ XGBoost ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ XGBoost : {e}")
        else:
            logger.warning("  ❌ XGBoost introuvable")

    def _load_tcn(self):
        p = _find_file(self.root,
            "outputs/NB06_V2/checkpoints/tcn_v3_final.pt",
            "outputs/NB06_V2/checkpoints/tcn_v2_final.pt",
            "outputs/NB06/checkpoints/tcn_final.pt",
        )
        if p:
            try:
                ckpt = torch.load(p, map_location=self.device, weights_only=False)
                nf   = ckpt.get("n_features", self._n_features)
                model = TCNClassifier(n_features=nf, n_channels=32,
                                      n_layers=3, kernel_size=3, dropout=0.3)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._tcn = model
                self._meta["tcn_threshold"] = ckpt.get("threshold", 0.5)
                self._meta["tcn_window"]    = ckpt.get("window_size", 6)
                logger.info(f"  ✅ TCN ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ TCN : {e}")
        else:
            logger.warning("  ❌ TCN introuvable")

    def _load_transformer(self):
        p = _find_file(self.root,
            "outputs/NB06_V2/checkpoints/transformer_v3_final.pt",
            "outputs/NB06_V2/checkpoints/transformer_v2_final.pt",
            "outputs/NB06/checkpoints/transformer_final.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                nf    = ckpt.get("n_features", self._n_features)
                model = TransformerClassifier(n_features=nf, d_model=48, n_heads=4,
                                              n_layers=2, d_ff=192, dropout=0.25)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._transformer = model
                self._meta["tf_threshold"] = ckpt.get("threshold", 0.5)
                logger.info(f"  ✅ Transformer ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ Transformer : {e}")
        else:
            logger.warning("  ❌ Transformer introuvable")

    def _load_lstm_ae(self):
        p = _find_file(self.root,
            "outputs/NB07/checkpoints/lstm_ae_final.pt",
            "outputs/NB07/checkpoints/lstm_ae_best.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                nf    = self._n_features
                model = LSTMAutoencoder(n_features=nf, hidden_dim=64, latent_dim=32,
                                        n_layers=2, seq_len=1, dropout=0.2)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._lstm_ae  = model
                self._thr_ae   = float(ckpt.get("threshold", 0.25))
                logger.info(f"  ✅ LSTM-AE ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ LSTM-AE : {e}")
        else:
            logger.warning("  ❌ LSTM-AE introuvable")

    def _load_vae(self):
        p = _find_file(self.root,
            "outputs/NB07/checkpoints/vae_final.pt",
            "outputs/NB07/checkpoints/vae_best.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                nf    = self._n_features
                model = VAE(n_features=nf, hidden_dims=(128, 64), latent_dim=16)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._vae      = model
                self._thr_vae  = float(ckpt.get("threshold", 48.0))
                logger.info(f"  ✅ VAE ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ VAE : {e}")
        else:
            logger.warning("  ❌ VAE introuvable")

    def _load_nhits(self):
        p = _find_file(self.root,
            "outputs/NB08_V2/checkpoints/nhits_v2_final.pt",
            "outputs/NB08_V2/checkpoints/nhits_v2_best.pt",
            "outputs/NB08/checkpoints/nhits_best.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                w     = ckpt.get("window", 12)
                h     = ckpt.get("horizon", 1)
                model = NHitsV2(window=w, horizon=h, n_hidden=32, dropout=0.2)
                state = ckpt["model_state"] if "model_state" in ckpt else ckpt
                model.load_state_dict(state)
                model.eval()
                self._nhits       = model
                self._nhits_mu    = float(ckpt.get("mu_train", 0.0))
                self._nhits_sigma = float(ckpt.get("sig_train", 1.0))
                self._meta["nhits_window"] = w
                logger.info(f"  ✅ N-HiTS ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ N-HiTS : {e}")
        else:
            logger.warning("  ❌ N-HiTS introuvable")

    def _load_tft(self):
        p = _find_file(self.root,
            "outputs/NB08_V2/checkpoints/tft_v2_final.pt",
            "outputs/NB08_V2/checkpoints/tft_v2_best.pt",
            "outputs/NB08/checkpoints/tft_best.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                w     = ckpt.get("window", 12)
                h     = ckpt.get("horizon", 1)
                model = TFTSimplified(window=w, horizon=h, d_model=32, n_heads=2, dropout=0.15)
                state = ckpt["model_state"] if "model_state" in ckpt else ckpt
                model.load_state_dict(state)
                model.eval()
                self._tft       = model
                self._tft_mu    = float(ckpt.get("mu_train", 0.0))
                self._tft_sigma = float(ckpt.get("sig_train", 1.0))
                self._meta["tft_window"] = w
                logger.info(f"  ✅ TFT ({p.name})")
            except Exception as e:
                logger.error(f"  ❌ TFT : {e}")
        else:
            logger.warning("  ❌ TFT introuvable")

    def _load_ppo(self):
        p = _find_file(self.root,
            "outputs/NB09/checkpoints/ppo_final.pt",
            "outputs/NB09/checkpoints/ppo_best.pt",
        )
        if p:
            try:
                ckpt  = torch.load(p, map_location=self.device, weights_only=False)
                obs_dim = int(ckpt.get("obs_dim", 216))
                model   = ActorCritic(obs_dim=obs_dim, n_actions=4, hidden=(256, 128))
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self._ppo             = model
                self._meta["ppo_obs_dim"] = obs_dim
                logger.info(f"  ✅ PPO ({p.name}, obs_dim={obs_dim})")
            except Exception as e:
                logger.error(f"  ❌ PPO : {e}")
        else:
            logger.warning("  ❌ PPO introuvable")

    def _init_shap(self):
        """Initialise le SHAP TreeExplainer après chargement XGBoost.

        Bug connu : shap.TreeExplainer(XGBClassifier) déclenche une erreur
        "could not convert string to float: '[5.653976E-1]'" car le wrapper
        sklearn appelle Booster.predict() qui renvoie une string en notation
        scientifique entre crochets dans certaines versions xgboost/shap.

        Fixes appliqués :
        1. On passe get_booster() (Booster natif) au lieu du XGBClassifier.
        2. Si le modèle n'a pas get_booster() (GradientBoostingClassifier sklearn),
           on utilise feature_perturbation='tree_path_dependent' comme fallback.
        3. En dernier recours, fallback sur KernelExplainer (plus lent mais fiable).
        """
        if self._xgb is None:
            return
        try:
            import shap

            # Stratégie 1 (priorité absolue) : Booster re-sauvegardé en JSON natif
            # _xgb_booster est créé par _load_xgboost via save_model/load_model,
            # ce qui efface le mismatch de version pkl et supprime le bug string-float
            booster = getattr(self, "_xgb_booster", None)
            if booster is not None:
                self._shap_explainer = shap.TreeExplainer(booster)
                logger.info("  ✅ SHAP TreeExplainer initialisé (Booster JSON natif)")
                return

            # Stratégie 2 : XGBClassifier → extraire le Booster à la volée
            if hasattr(self._xgb, "get_booster"):
                self._shap_explainer = shap.TreeExplainer(self._xgb.get_booster())
                logger.info("  ✅ SHAP TreeExplainer initialisé (get_booster)")
                return

            # Stratégie 3 : sklearn tree model (GradientBoostingClassifier, etc.)
            self._shap_explainer = shap.TreeExplainer(
                self._xgb,
                feature_perturbation="tree_path_dependent",
            )
            logger.info("  ✅ SHAP TreeExplainer initialisé (sklearn tree)")

        except ImportError:
            logger.warning("  ⚠️  shap non installé — pip install shap")
        except Exception as e:
            # Stratégie 4 : KernelExplainer — modèle-agnostique, garanti compatible
            logger.warning(f"  ⚠️  TreeExplainer échoué ({e}) → KernelExplainer")
            try:
                import shap, numpy as np
                n_feat = getattr(self, "_n_features", 44)
                # bg = np.zeros((1, n_feat), dtype=np.float32)
                rng = np.random.default_rng(42)
                bg = rng.uniform(0.01, 1.0, size=(10, n_feat)).astype(np.float32)
                predict_fn = (self._xgb.predict_proba
                              if hasattr(self._xgb, "predict_proba")
                              else self._xgb.predict)
                self._shap_explainer = shap.KernelExplainer(predict_fn, bg)
                self._shap_mode = "kernel"
                logger.info("  ✅ SHAP KernelExplainer initialisé (fallback)")
            except Exception as e2:
                logger.warning(f"  ⚠️  SHAP totalement désactivé : {e2}")

    # ─── Normalisation ───────────────────────────────────────────────────────

    def _scale(self, vector: List[float]) -> np.ndarray:
        """Applique RobustScaler sur un vecteur de features."""
        x = np.array([vector], dtype=np.float32)
        if self._scaler is not None:
            try:
                return self._scaler.transform(x)[0]
            except Exception:
                pass
        return x[0]

    def _build_sequence(self, vector_scaled: np.ndarray, window: int = 6) -> np.ndarray:
        """Répète le vecteur normalisé pour former une fenêtre temporelle (B, W, F)."""
        return np.tile(vector_scaled, (window, 1))

    # ─── Prédiction XGBoost ──────────────────────────────────────────────────

    def _predict_xgb(self, vector: np.ndarray) -> float:
        """Retourne prob_panne ∈ [0,1] depuis XGBoost."""
        if self._xgb is None:
            return 0.0
        try:
            proba = self._xgb.predict_proba(vector.reshape(1, -1))
            return float(proba[0, 1])
        except Exception as e:
            logger.debug(f"XGBoost predict error: {e}")
            return 0.0

    def _predict_tcn(self, vector_scaled: np.ndarray, window: int = 6) -> float:
        """Retourne prob_panne TCN ∈ [0,1]."""
        if self._tcn is None:
            return 0.0
        try:
            seq = self._build_sequence(vector_scaled, window)
            X   = torch.tensor([seq], dtype=torch.float32)
            with torch.no_grad():
                logit = self._tcn(X)
                return float(torch.sigmoid(logit).item())
        except Exception as e:
            logger.debug(f"TCN predict error: {e}")
            return 0.0

    def _predict_anomaly(self, vector_scaled: np.ndarray) -> Tuple[float, float]:
        """Retourne (score_ae_norm, score_vae_norm) ∈ [0,1]."""
        score_ae = score_vae = 0.0

        if self._lstm_ae is not None:
            try:
                X   = torch.tensor([vector_scaled], dtype=torch.float32).unsqueeze(1)
                raw = float(self._lstm_ae.reconstruction_error(X)[0])
                # Normaliser par rapport au seuil
                score_ae = min(raw / max(self._thr_ae, 1e-6), 5.0) / 5.0
            except Exception as e:
                logger.debug(f"LSTM-AE predict error: {e}")

        if self._vae is not None:
            try:
                X   = torch.tensor([vector_scaled], dtype=torch.float32)
                raw = float(self._vae.anomaly_score(X)[0])
                score_vae = min(raw / max(self._thr_vae, 1e-6), 5.0) / 5.0
            except Exception as e:
                logger.debug(f"VAE predict error: {e}")

        return score_ae, score_vae

    def _predict_forecast(self, carb_history: List[float], model_key: str = "nhits",
                          horizon: int = 3) -> List[float]:
        """Retourne les prévisions carburant pour `horizon` mois."""
        model  = self._nhits if model_key == "nhits" else self._tft
        mu     = self._nhits_mu    if model_key == "nhits" else self._tft_mu
        sigma  = self._nhits_sigma if model_key == "nhits" else self._tft_sigma
        window = int(self._meta.get(f"{model_key}_window", 12))

        if model is None or not carb_history:
            return [0.0] * horizon

        # Repli mu/sigma sur les stats de l'historique si checkpoint sans mu/sigma
        if abs(mu) < 1e-6 and abs(sigma - 1.0) < 1e-6:
            mu    = float(np.mean(carb_history))
            sigma = float(np.std(carb_history)) or 1.0

        try:
            hist_norm = [(v - mu) / sigma for v in carb_history]
            series    = list(hist_norm[-window:])
            if len(series) < window:
                series = [series[0]] * (window - len(series)) + series if series else [0.0] * window

            preds_norm = []
            for _ in range(horizon):
                x   = torch.tensor([series[-window:]], dtype=torch.float32)
                with torch.no_grad():
                    out = model(x)
                    p   = float(out.flatten()[0].item())
                preds_norm.append(p)
                series.append(p)

            return [round(p * sigma + mu, 1) for p in preds_norm]
        except Exception as e:
            logger.debug(f"Forecast ({model_key}) error: {e}")
            return [0.0] * horizon

    def _predict_ppo(self, vehicle_data: dict, fleet_context: Optional[dict] = None) -> Tuple[int, float, dict]:
        """
        Reconstruit l'observation de flotte (216 dims) et retourne
        (action_index, confidence, probs_dict).
        """
        if self._ppo is None:
            return 0, 0.0, {}

        obs_dim = int(self._meta.get("ppo_obs_dim", 216))
        n_vehs  = obs_dim // 8

        km_total_6m = float(vehicle_data.get("km_total_6m", 0))
        cout_6m     = float(vehicle_data.get("cout_intervention_6m", 0))
        n_interv    = float(vehicle_data.get("nb_interventions_6m", 0))
        mois_dep    = float(vehicle_data.get("mois_depuis_debut", 0))
        mois_act    = int(vehicle_data.get("mois_actuel", 6))

        km_norm    = min(km_total_6m / 60000, 1.0)
        cout_norm  = min(cout_6m / 50000, 1.0)
        interv_norm= min(n_interv / 5, 1.0)
        maint_norm = min(mois_dep / 12, 1.0)
        degr       = min(km_norm * 0.6 + interv_norm * 0.4, 1.0)
        panne_p    = min(0.04 + 0.5 * degr ** 2, 0.8)
        t_norm     = 0.5
        saison     = math.sin(2 * math.pi * mois_act / 12)

        veh_features  = [km_norm, cout_norm, interv_norm, maint_norm,
                         degr, panne_p, t_norm, saison]
        fleet_normal  = [0.1, 0.05, 0.0, 0.3, 0.05, 0.04, 0.5, saison]
        obs_list = veh_features + fleet_normal * (n_vehs - 1)
        obs_list = obs_list[:obs_dim]

        try:
            obs   = torch.tensor([obs_list], dtype=torch.float32)
            probs = self._ppo.get_action_probs(obs)[0]
            best  = int(np.argmax(probs))
            probs_dict = {ACTIONS[i]: float(probs[i]) for i in range(len(ACTIONS))}
            return best, float(probs[best]), probs_dict
        except Exception as e:
            logger.debug(f"PPO predict error: {e}")
            return 0, 0.0, {}

    def _compute_shap(self, vector: np.ndarray) -> Dict[str, float]:
        """Calcule les valeurs SHAP pour un vecteur de features brutes (non normalisé)."""
        if self._shap_explainer is None or not self._feature_cols:
            return {}
        try:
            raw = self._shap_explainer.shap_values(vector.reshape(1, -1))

            # Normaliser le format de sortie selon le type d'explainer :
            #
            # TreeExplainer (Booster natif) → np.ndarray shape (1, n_feat)
            # TreeExplainer (sklearn)       → list[array_neg, array_pos] (binaire)
            # KernelExplainer               → list[array_neg, array_pos]
            # shap.Explanation              → objet avec .values
            if hasattr(raw, "values"):
                # shap >= 0.40 retourne un objet Explanation
                arr = np.asarray(raw.values, dtype=float).reshape(-1)
            elif isinstance(raw, list):
                # Classification binaire : [classe_0, classe_1] → prendre classe 1
                arr = np.asarray(raw[1], dtype=float).reshape(-1)
            else:
                # ndarray (2D ou 1D)
                arr = np.asarray(raw, dtype=float).reshape(-1)

            return {self._feature_cols[i]: round(float(arr[i]), 4)
                    for i in range(min(len(arr), len(self._feature_cols)))}
        except Exception as e:
            logger.warning(f"SHAP compute error: {e}")
            return {}

    @staticmethod
    def _compute_criticite(prob_panne: float, score_anomalie: float,
                           forecast_ecart: float) -> float:
        """Criticité composite : 0.5×panne + 0.3×anomalie + 0.2×écart prévision."""
        return min(0.5 * prob_panne + 0.3 * score_anomalie + 0.2 * forecast_ecart, 1.0)

    @staticmethod
    def _priorite(criticite: float) -> str:
        if criticite >= 0.75:  return "CRITIQUE"
        if criticite >= 0.50:  return "HAUTE"
        if criticite >= 0.25:  return "MOYENNE"
        return "FAIBLE"

    @staticmethod
    def _build_explications(prob_panne: float, score_ae: float,
                            action: str, top_features: List[dict]) -> List[str]:
        """Génère des phrases d'explication lisibles par un technicien."""
        lines = []
        if prob_panne >= 0.6:
            lines.append(f"⚠️ Probabilité de panne élevée ({prob_panne*100:.0f}%) — maintenance recommandée.")
        elif prob_panne >= 0.35:
            lines.append(f"🔶 Risque de panne modéré ({prob_panne*100:.0f}%) — surveiller.")
        else:
            lines.append(f"✅ Faible probabilité de panne ({prob_panne*100:.0f}%).")

        if score_ae >= 0.5:
            lines.append("🔴 Comportement anormal détecté par le modèle non-supervisé.")

        if top_features:
            f1 = top_features[0]
            lines.append(f"📊 Feature la plus influente : {f1['name']} (SHAP={f1['value']:+.3f}).")

        lines.append(f"🤖 PPO recommande : {action}.")
        return lines

    # ─── Interface publique ───────────────────────────────────────────────────

    def predict_vehicle(self, vehicle_data: dict,
                        carb_history: Optional[List[float]] = None) -> AIVehicleReport:
        """
        Calcule le rapport IA complet pour un véhicule.

        Parameters
        ----------
        vehicle_data : dict
            Champs attendus (mêmes que le formulaire Flask) :
              immat, vehicule_id, km_total_6m, carburant_total_6m,
              nb_pleins_6m, nb_interventions_6m, cout_intervention_6m,
              mois_depuis_debut, mois_actuel, + toutes les features NB-04.
        carb_history : list[float], optional
            Historique mensuel carburant (12 valeurs idéalement).
        """
        immat       = str(vehicle_data.get("immat", "UNKNOWN"))
        vehicule_id = int(vehicle_data.get("vehicule_id", 0))

        # 1. Construire le vecteur de features brutes dans l'ordre du manifest
        if self._feature_cols:
            vector_raw = np.array(
                [float(vehicle_data.get(col, 0.0)) for col in self._feature_cols],
                dtype=np.float32,
            )
        else:
            # Fallback : 5 features de base
            vector_raw = np.array([
                float(vehicle_data.get("km_total_6m", 0)),
                float(vehicle_data.get("carburant_total_6m", 0)),
                float(vehicle_data.get("nb_interventions_6m", 0)),
                float(vehicle_data.get("cout_intervention_6m", 0)),
                float(vehicle_data.get("mois_depuis_debut", 0)),
            ], dtype=np.float32)

        # 2. Normaliser (RobustScaler)
        vector_scaled = self._scale(vector_raw)

        # 3. XGBoost — prob panne (features brutes)
        prob_panne = self._predict_xgb(vector_raw[:self._n_features]
                                       if len(vector_raw) >= self._n_features
                                       else vector_raw)

        # 4. TCN — prob panne séquentielle (features normalisées)
        tcn_window    = int(self._meta.get("tcn_window", 6))
        prob_panne_tcn= self._predict_tcn(vector_scaled, window=tcn_window)

        # 5. LSTM-AE / VAE — anomalies
        score_ae, score_vae = self._predict_anomaly(vector_scaled)
        score_anomalie      = max(score_ae, score_vae)
        anomalie_detectee   = score_anomalie >= 0.5

        # 6. N-HiTS — prévision carburant
        if carb_history is None:
            carb_mensuel = float(vehicle_data.get("carburant_total_6m", 0)) / 6
            carb_history = [carb_mensuel] * 12

        forecast    = self._predict_forecast(carb_history, model_key="nhits", horizon=3)
        hist_mean   = float(np.mean(carb_history)) if carb_history else 1.0
        fc_mean     = float(np.mean(forecast))     if forecast     else 0.0
        forecast_ecart = abs(fc_mean - hist_mean) / max(hist_mean, 1.0)  # fraction

        # 7. PPO — action recommandée
        action_idx, action_conf, action_probs = self._predict_ppo(vehicle_data)
        action_name = ACTIONS[action_idx] if action_idx < len(ACTIONS) else ACTIONS[0]

        # 8. SHAP
        shap_values = self._compute_shap(vector_raw)
        top_features = sorted(
            [{"name": k, "value": v, "abs": abs(v)} for k, v in shap_values.items()],
            key=lambda x: x["abs"], reverse=True
        )[:5]

        # 9. Criticité composite
        criticite = self._compute_criticite(prob_panne, score_anomalie,
                                            min(forecast_ecart, 1.0))
        priorite  = self._priorite(criticite)

        # 10. Explicabilité
        explications = self._build_explications(prob_panne, score_ae, action_name, top_features)

        # 11. Confiance globale (moyenne des modèles disponibles)
        confs = [prob_panne, prob_panne_tcn, score_anomalie, action_conf]
        conf_globale = float(np.mean([c for c in confs if c > 0])) if confs else 0.0

        modeles = []
        if self._xgb:        modeles.append("XGBoost")
        if self._tcn:        modeles.append("TCN")
        if self._lstm_ae:    modeles.append("LSTM-AE")
        if self._vae:        modeles.append("VAE")
        if self._nhits:      modeles.append("N-HiTS")
        if self._ppo:        modeles.append("PPO")

        return AIVehicleReport(
            immat              = immat,
            vehicule_id        = vehicule_id,
            prob_panne         = prob_panne,
            prob_panne_tcn     = prob_panne_tcn,
            score_anomalie_ae  = score_ae,
            score_anomalie_vae = score_vae,
            score_anomalie     = score_anomalie,
            anomalie_detectee  = anomalie_detectee,
            forecast_carburant = forecast,
            forecast_ecart_pct = round(forecast_ecart * 100, 1),
            action_recommandee = action_name,
            action_index       = action_idx,
            action_confidence  = action_conf,
            action_probs       = action_probs,
            criticite          = criticite,
            priorite           = priorite,
            shap_values        = shap_values,
            top_features       = top_features,
            explications       = explications,
            modeles_utilises   = modeles,
            confidence_globale = conf_globale,
        )

    def predict_fleet(self, vehicles: List[dict],
                      carb_histories: Optional[Dict[int, List[float]]] = None
                      ) -> List[AIVehicleReport]:
        """
        Calcule les rapports IA pour toute la flotte et trie par criticité décroissante.

        Parameters
        ----------
        vehicles : list[dict]
            Un dict par véhicule (format identique à predict_vehicle).
        carb_histories : dict[int, list[float]], optional
            Historique carburant indexé par vehicule_id.
        """
        if carb_histories is None:
            carb_histories = {}

        reports = []
        for v in vehicles:
            vid = int(v.get("vehicule_id", 0))
            hist = carb_histories.get(vid)
            try:
                report = self.predict_vehicle(v, carb_history=hist)
            except Exception as e:
                logger.error(f"Erreur prédiction véhicule {v.get('immat','?')}: {e}")
                report = AIVehicleReport(
                    immat=str(v.get("immat","?")),
                    vehicule_id=vid,
                    explications=[f"Erreur moteur IA : {e}"],
                )
            reports.append(report)

        # Trier par criticité décroissante
        reports.sort(key=lambda r: r.criticite, reverse=True)
        return reports

    def health(self) -> dict:
        """Retourne l'état de santé du moteur IA (pour GET /models/health)."""
        return {
            "ready":    self.ready,
            "device":   str(self.device),
            "models": {
                "scaler":      self._scaler      is not None,
                "xgboost":     self._xgb         is not None,
                "tcn":         self._tcn         is not None,
                "transformer": self._transformer is not None,
                "lstm_ae":     self._lstm_ae     is not None,
                "vae":         self._vae         is not None,
                "nhits":       self._nhits        is not None,
                "tft":         self._tft         is not None,
                "ppo":         self._ppo         is not None,
                "shap":        self._shap_explainer is not None,
            },
            "meta": self._meta,
            "n_features": self._n_features,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SINGLETON — instance partagée pour FastAPI (lifespan)
# ═══════════════════════════════════════════════════════════════════════════════

_engine_instance: Optional[AIEngine] = None


def get_engine() -> AIEngine:
    """Retourne l'instance singleton du moteur IA (à utiliser dans les routes FastAPI)."""
    global _engine_instance
    if _engine_instance is None:
        raise RuntimeError("AIEngine non initialisé — appeler init_engine() au démarrage")
    return _engine_instance


def init_engine(project_root: str, device: str = "cpu") -> AIEngine:
    """Initialise et charge le moteur IA (à appeler dans le lifespan FastAPI)."""
    global _engine_instance
    _engine_instance = AIEngine(project_root=project_root, device=device)
    _engine_instance.load()
    return _engine_instance


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TEST RAPIDE (python ai_engine.py)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    engine = AIEngine(project_root=root)
    engine.load()

    print("\n" + "="*60)
    print("  TEST — Profil SAIN")
    print("="*60)
    sain = {
        "immat": "AMB-001", "vehicule_id": 1,
        "km_total_6m": 3000, "carburant_total_6m": 250, "nb_pleins_6m": 6,
        "nb_interventions_6m": 0, "cout_intervention_6m": 0,
        "mois_depuis_debut": 12, "mois_actuel": 6,
    }
    r1 = engine.predict_vehicle(sain, carb_history=[40,42,38,45,41,39,43,40,44,42,38,41])
    print(f"  Criticité   : {r1.criticite:.3f}  → {r1.priorite}")
    print(f"  Prob panne  : {r1.prob_panne:.3f}")
    print(f"  Anomalie    : {r1.score_anomalie:.3f} (détectée={r1.anomalie_detectee})")
    print(f"  Forecast    : {r1.forecast_carburant}")
    print(f"  Action PPO  : {r1.action_recommandee} (conf={r1.action_confidence:.2f})")
    for e in r1.explications:
        print(f"  {e}")

    print("\n" + "="*60)
    print("  TEST — Profil A RISQUE")
    print("="*60)
    risque = {
        "immat": "AMB-099", "vehicule_id": 99,
        "km_total_6m": 9500, "carburant_total_6m": 720, "nb_pleins_6m": 18,
        "nb_interventions_6m": 5, "cout_intervention_6m": 15000,
        "mois_depuis_debut": 84, "mois_actuel": 6,
    }
    r2 = engine.predict_vehicle(risque, carb_history=[95,110,88,130,105,140,98,125,115,135,102,145])
    print(f"  Criticité   : {r2.criticite:.3f}  → {r2.priorite}")
    print(f"  Prob panne  : {r2.prob_panne:.3f}")
    print(f"  Anomalie    : {r2.score_anomalie:.3f} (détectée={r2.anomalie_detectee})")
    print(f"  Forecast    : {r2.forecast_carburant}")
    print(f"  Action PPO  : {r2.action_recommandee} (conf={r2.action_confidence:.2f})")
    for e in r2.explications:
        print(f"  {e}")

    print("\n  Health check :")
    h = engine.health()
    for name, ok in h["models"].items():
        print(f"    {'✅' if ok else '❌'} {name}")