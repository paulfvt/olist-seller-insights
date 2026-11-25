"""Client LLM unique pour Groq (par défaut) et Mistral, avec cache et gestion des quotas.

Ce qu'est un appel LLM ici
--------------------------
On envoie une liste de messages (rôle "system" = consignes, rôle "user" = la demande) à
une API HTTP ; le modèle renvoie un texte. Mistral et Groq exposent tous deux le même
format, dit « compatible OpenAI » (POST /chat/completions). Un seul code suffit donc :
seuls l'URL, la clé et le nom du modèle changent. On évite ainsi deux SDK différents.

Respect des quotas gratuits
---------------------------
Les offres gratuites limitent le nombre de requêtes par minute et de tokens par jour.
- Cache : une réponse est stockée sous une clé (fournisseur, modèle, messages,
  température). Relancer le pipeline ne refait aucun appel déjà fait.
- Intervalle minimal entre deux appels (LLM_MIN_INTERVAL_S) : on ne déclenche pas la
  limite par excès de vitesse.
- Backoff exponentiel : si l'API répond 429 (trop de requêtes) ou 5xx (panne passagère),
  on attend 2 s, puis 4 s, 8 s... (+ un peu d'aléatoire, le « jitter », pour que des
  clients en échec ne réessaient pas tous au même instant). Si l'API indique elle-même
  combien attendre (en-tête Retry-After), on lui obéit.
Les clés API viennent de .env (jamais commité) et n'apparaissent jamais dans les logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import duckdb
import httpx

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    model_env: str
    default_model: str


# Les noms de modèles se changent dans .env, pas dans le code (ils évoluent souvent).
# Groq est le fournisseur par défaut : en novembre 2026, l'API Mistral exige l'achat de
# crédits prépayés (constaté sur la console), incompatible avec la contrainte « 100 % gratuit ».
# Modèle Groq : gpt-oss-120b. Comparé sur les 10 vendeurs les plus à risque (prompt v3) :
# 10/10 explications valides contre 5/10 pour gpt-oss-20b (JSON souvent rejeté).
DEFAULT_PROVIDER = "groq"
PROVIDERS = {
    "groq": ProviderConfig("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
                           "GROQ_MODEL", "openai/gpt-oss-120b"),
    "mistral": ProviderConfig("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                              "MISTRAL_MODEL", "mistral-small-latest"),
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Échec définitif d'un appel (clé invalide, quota épuisé après tous les essais...)."""


class LLMOutputError(LLMError):
    """Le modèle a produit une sortie rejetée par le fournisseur (ex. JSON invalide en mode
    JSON : Groq répond 400 'json_validate_failed'). Ponctuel : un nouvel essai peut réussir,
    et cela ne concerne que cette requête, pas la configuration."""


class LLMClient(Protocol):
    """Contrat commun : le reste du code ne dépend que de ceci (et se teste avec un faux)."""

    provider: str
    model: str

    def complete(self, messages: list[Message], json_mode: bool = True) -> str: ...


# ---------------------------------------------------------------------------
# Cache des réponses
# ---------------------------------------------------------------------------


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_responses (
                key VARCHAR PRIMARY KEY, provider VARCHAR, model VARCHAR,
                request JSON, response VARCHAR, created_at TIMESTAMP
            )
            """
        )

    @staticmethod
    def key(provider: str, model: str, messages: list[Message], temperature: float, json_mode: bool) -> str:
        payload = json.dumps([provider, model, messages, temperature, json_mode], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.con.execute("SELECT response FROM llm_responses WHERE key = ?", [key]).fetchone()
        return row[0] if row else None

    def put(self, key: str, provider: str, model: str, messages: list[Message], response: str) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.con.execute(
            "INSERT INTO llm_responses VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [key, provider, model, json.dumps(messages, ensure_ascii=False), response, now],
        )

    def close(self) -> None:
        self.con.close()


# ---------------------------------------------------------------------------
# Client HTTP compatible OpenAI
# ---------------------------------------------------------------------------


def backoff_delay(attempt: int, retry_after: str | None, base: float = 2.0, cap: float = 60.0) -> float:
    """Attente avant l'essai n° attempt+1 (attempt commence à 0).

    Retry-After (en secondes) prime s'il est fourni ; sinon base * 2^attempt, plafonné,
    multiplié par un facteur aléatoire entre 0,5 et 1 (jitter).
    """
    if retry_after:
        try:
            return min(float(retry_after), cap)
        except ValueError:
            pass  # Retry-After peut aussi être une date HTTP : on retombe sur le calcul
    return min(cap, base * 2**attempt) * random.uniform(0.5, 1.0)


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        cache: ResponseCache | None = None,
        temperature: float = 0.0,
        min_interval_s: float = 1.5,
        max_retries: int = 5,
        timeout_s: float = 60.0,
        http: httpx.Client | None = None,
        sleep=time.sleep,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cache = cache
        # Température 0 : le modèle choisit toujours le token le plus probable, réponses
        # les plus stables possibles d'un appel à l'autre (utile pour le cache et l'évaluation).
        self.temperature = temperature
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._sleep = sleep
        self._last_call = 0.0
        self._http = http or httpx.Client(
            base_url=base_url, timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.n_api_calls = 0
        self.n_cache_hits = 0

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def complete(self, messages: list[Message], json_mode: bool = True) -> str:
        key = ResponseCache.key(self.provider, self.model, messages, self.temperature, json_mode)
        if self.cache and (cached := self.cache.get(key)) is not None:
            self.n_cache_hits += 1
            return cached

        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if json_mode:
            # Mode JSON : le fournisseur garantit une sortie JSON syntaxiquement valide
            # (mais pas qu'elle respecte NOTRE schéma : on le vérifie nous-mêmes ensuite).
            body["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:  # coupure réseau, délai dépassé
                status, retry_after, detail = None, None, type(exc).__name__
            else:
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    self.n_api_calls += 1
                    if self.cache:
                        self.cache.put(key, self.provider, self.model, messages, text)
                    return text
                status, retry_after, detail = resp.status_code, resp.headers.get("retry-after"), resp.text[:200]
                if status == 429 and resp.headers.get("x-ratelimit-limit-req-minute") == "0":
                    # Quota de 0 requête/minute : ce n'est pas un excès de vitesse mais un compte
                    # sans accès API (offre non activée). Attendre ne servira à rien.
                    raise LLMError(f"{self.provider} : quota de 0 requête/minute sur ce compte "
                                   "(offre gratuite non activée ?) - vérifier la console du fournisseur")
                if status == 400 and "json_validate_failed" in resp.text:
                    raise LLMOutputError(f"{self.provider} : le modèle a produit un JSON invalide")
                if status not in RETRYABLE_STATUS:
                    # 400 (requête invalide), 401 (clé refusée)... réessayer ne changera rien
                    raise LLMError(f"{self.provider} HTTP {status} : {detail}")
            if attempt == self.max_retries:
                break
            delay = backoff_delay(attempt, retry_after)
            logger.warning("%s : %s %s, nouvel essai dans %.1f s (%d/%d)",
                           self.provider, status or "", detail[:80], delay, attempt + 1, self.max_retries)
            self._sleep(delay)
        raise LLMError(f"{self.provider} : échec après {self.max_retries + 1} essais ({status or detail})")


def make_client(provider: str | None = None, cache_path: Path | None = None) -> OpenAICompatibleClient:
    """Construit le client du fournisseur choisi (LLM_PROVIDER dans .env, par défaut groq)."""
    name = (provider or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)).lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER inconnu : {name!r} (attendu : {', '.join(PROVIDERS)})")
    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise LLMError(f"{cfg.api_key_env} absente : ajoute ta clé dans le fichier .env (voir .env.example)")
    return OpenAICompatibleClient(
        provider=name,
        model=os.getenv(cfg.model_env, cfg.default_model),
        api_key=api_key,
        base_url=cfg.base_url,
        cache=ResponseCache(cache_path) if cache_path else None,
        min_interval_s=float(os.getenv("LLM_MIN_INTERVAL_S", "1.5")),
    )
