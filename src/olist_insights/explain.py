"""Explication des difficultés d'un vendeur par un LLM, à partir de ses KPI et de ses avis.

Pourquoi tant de contrôles ?
----------------------------
Un LLM produit du texte plausible, pas du texte vrai : il peut inventer un chiffre
(« 35 % de retards ») ou un avis client qui n'existe pas. On ne lui fait donc pas
confiance, on VÉRIFIE :
1. on lui fournit les seuls chiffres autorisés (liste {kpi_name: valeur}) et des avis
   numérotés R1, R2... ;
2. on exige un JSON structuré où chaque cause cite ses KPI (nom + valeur recopiée) et
   ses avis (référence + citation exacte) ;
3. on contrôle tout automatiquement : schéma (pydantic), KPI existants et valeurs
   identiques, avis existants, citations réellement présentes dans l'avis, aucun nombre
   dans le texte libre qui ne soit pas un KPI fourni ;
4. en cas d'erreur, un seul nouvel essai en renvoyant la liste des erreurs au modèle
   (il se corrige souvent) ; sinon la réponse est marquée invalide, jamais acceptée en silence.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb
from pydantic import BaseModel, Field, ValidationError

from olist_insights.config import load_settings
from olist_insights.llm_client import LLMClient, LLMError, LLMOutputError, Message, make_client
from olist_insights.logging_utils import setup_logging

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3"  # à incrémenter à chaque changement de consigne (le cache en dépend)

# KPI du vendeur -> (KPI de référence marketplace, True si « plus haut = pire »).
# Un KPI ne peut appuyer une cause que s'il est PIRE que la marketplace : un chiffre exact
# peut servir une conclusion fausse (« 6,5 % de retards » présenté comme un problème
# alors que la moyenne est 6,9 %). Ce contrôle-là, la vérification des valeurs ne le voit pas.
BENCHMARKS: dict[str, tuple[str, bool]] = {
    "late_delivery_rate_pct": ("marketplace_late_delivery_rate_pct", True),
    "last_90d_late_delivery_rate_pct": ("marketplace_late_delivery_rate_pct", True),
    "late_carrier_handoff_rate_pct": ("marketplace_late_carrier_handoff_rate_pct", True),
    "cancel_rate_pct": ("marketplace_cancel_rate_pct", True),
    "avg_rating": ("marketplace_avg_rating", False),
    "last_90d_avg_rating": ("marketplace_avg_rating", False),
    "low_rating_rate_pct": ("marketplace_low_rating_rate_pct", True),
    "last_90d_low_rating_rate_pct": ("marketplace_low_rating_rate_pct", True),
}
NOT_EVIDENCE = {"risk_score_pct"}  # sortie du modèle : la citer comme cause serait circulaire
MAX_NEGATIVE_REVIEWS = 15
MAX_OTHER_REVIEWS = 8
LLM_CACHE_FILENAME = "llm_responses.duckdb"


# ---------------------------------------------------------------------------
# Schéma de sortie attendu
# ---------------------------------------------------------------------------


class KPIRef(BaseModel):
    kpi_name: str
    value: float


class ReviewRef(BaseModel):
    review_ref: str = Field(pattern=r"^R\d+$")
    quote: str = Field(min_length=3)


class Cause(BaseModel):
    title: str
    explanation: str
    supporting_kpis: list[KPIRef]
    supporting_reviews: list[ReviewRef] = Field(min_length=1)  # une cause sans avis = spéculation


class Improvement(BaseModel):
    action: str
    cause_index: int = Field(ge=0)  # index de la cause traitée (0 = première)
    rationale: str


class Explanation(BaseModel):
    summary: str
    # 0 cause autorisé : si aucun KPI n'est pire que la marketplace, le score de risque est
    # peut-être un faux positif du modèle, et le LLM doit pouvoir le dire au lieu d'inventer.
    causes: list[Cause] = Field(default_factory=list, max_length=4)
    improvements: list[Improvement] = Field(default_factory=list, max_length=5)


# ---------------------------------------------------------------------------
# Contexte : KPI autorisés + échantillon d'avis
# ---------------------------------------------------------------------------


@dataclass
class SellerContext:
    seller_id: str
    facts: dict[str, float]                          # seuls chiffres que le LLM peut citer
    reviews: dict[str, dict] = field(default_factory=dict)  # "R1" -> {review_id, score, text}


def _pct(x) -> float | None:
    return None if x is None else round(100 * float(x), 1)


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    (n,) = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? AND table_name = ?", [schema, table]
    ).fetchone()
    return n > 0


def load_facts(con: duckdb.DuckDBPyConnection, seller_id: str) -> dict[str, float]:
    """KPI du vendeur, arrondis comme ils seront affichés, + moyennes de la marketplace
    pour comparer (un taux de retard de 10 % ne veut rien dire sans référence)."""
    row = con.execute(
        """
        SELECT k.*,
               avg(late_rate) OVER ()       AS mkt_late_rate,
               avg(cancel_rate) OVER ()     AS mkt_cancel_rate,
               avg(late_handoff_rate) OVER () AS mkt_late_handoff_rate,
               avg(avg_rating) OVER ()      AS mkt_avg_rating,
               avg(low_rating_rate) OVER () AS mkt_low_rating_rate
        FROM gold.seller_kpis k
        QUALIFY seller_id = ?
        """,
        [seller_id],
    ).df()
    if row.empty:
        raise KeyError(f"vendeur inconnu : {seller_id}")
    k = row.iloc[0]
    facts = {
        "n_orders": int(k["n_orders"]),
        "revenue_brl": round(float(k["revenue"])),
        "cancel_rate_pct": _pct(k["cancel_rate"]),
        "late_delivery_rate_pct": _pct(k["late_rate"]),
        "avg_delay_vs_estimate_days": None if k["avg_delay_days"] != k["avg_delay_days"] else round(float(k["avg_delay_days"]), 1),
        "avg_delivery_days": None if k["avg_delivery_days"] != k["avg_delivery_days"] else round(float(k["avg_delivery_days"]), 1),
        "late_carrier_handoff_rate_pct": _pct(k["late_handoff_rate"]),
        "freight_to_price_ratio_pct": _pct(k["freight_ratio"]),
        "n_reviews": int(k["n_reviews"]),
        "avg_rating": None if k["avg_rating"] != k["avg_rating"] else round(float(k["avg_rating"]), 2),
        "low_rating_rate_pct": _pct(k["low_rating_rate"]),
        "marketplace_late_delivery_rate_pct": _pct(k["mkt_late_rate"]),
        "marketplace_cancel_rate_pct": _pct(k["mkt_cancel_rate"]),
        "marketplace_late_carrier_handoff_rate_pct": _pct(k["mkt_late_handoff_rate"]),
        "marketplace_avg_rating": round(float(k["mkt_avg_rating"]), 2),
        "marketplace_low_rating_rate_pct": _pct(k["mkt_low_rating_rate"]),
    }
    if _table_exists(con, "gold", "seller_risk"):
        risk = con.execute(
            "SELECT risk_score, avg_rating_90d, low_rate_90d, late_rate_90d FROM gold.seller_risk WHERE seller_id = ?",
            [seller_id],
        ).fetchone()
        if risk:
            facts.update({
                "risk_score_pct": _pct(risk[0]),
                "last_90d_avg_rating": round(float(risk[1]), 2),
                "last_90d_low_rating_rate_pct": _pct(risk[2]),
                "last_90d_late_delivery_rate_pct": _pct(risk[3]),
            })
    return {name: value for name, value in facts.items() if value is not None}


def load_reviews(con: duckdb.DuckDBPyConnection, seller_id: str) -> dict[str, dict]:
    """Avis traduits : les négatifs les plus récents (ils expliquent les causes) puis
    quelques autres, choisis de façon déterministe (ordre par empreinte de l'identifiant)."""
    rows = con.execute(
        f"""
        WITH r AS (
            SELECT s.review_id, s.review_score AS score, s.review_created_at, t.review_en AS text
            FROM silver.seller_reviews s
            JOIN silver.review_translations t USING (review_id)
            WHERE s.seller_id = ? AND s.has_text AND t.review_en IS NOT NULL
        ),
        neg AS (SELECT * FROM r WHERE score <= 3 ORDER BY review_created_at DESC, review_id LIMIT {MAX_NEGATIVE_REVIEWS}),
        other AS (SELECT * FROM r WHERE score >= 4 ORDER BY md5(review_id) LIMIT {MAX_OTHER_REVIEWS})
        SELECT review_id, score, text FROM (SELECT * FROM neg UNION ALL SELECT * FROM other)
        ORDER BY score, review_id
        """,
        [seller_id],
    ).fetchall()
    return {f"R{i}": {"review_id": rid, "score": int(score), "text": text}
            for i, (rid, score, text) in enumerate(rows, start=1)}


def load_context(con: duckdb.DuckDBPyConnection, seller_id: str) -> SellerContext:
    return SellerContext(seller_id, load_facts(con, seller_id), load_reviews(con, seller_id))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an analyst for an e-commerce marketplace. You explain why a seller's \
customer satisfaction is at risk and suggest concrete improvements.

Strict rules:
1. Use ONLY the facts and reviews provided. Never invent a number, a review or an event.
2. Any number you mention must be one of the provided KPI values, copied exactly. \
Do not compute new numbers (no differences, ratios or rounding).
3. Every cause must cite at least one KPI and at least one review. For each review, give its \
reference (e.g. "R3") and an exact quote copied word for word from that review's text.
4. If the reviews do not support a cause, do not mention that cause.
5. Compare the seller with the marketplace_* values. A KPI can support a cause only if it is \
worse than its marketplace value (higher for rates, lower for ratings); cite both values. \
risk_score_pct is a model prediction, not evidence: never cite it in supporting_kpis.
6. If no KPI is worse than the marketplace, return empty "causes" and "improvements" lists and \
say in the summary that the data does not confirm a satisfaction problem.
7. Answer with a single JSON object, no text around it, with this structure:
{"summary": str,
 "causes": [{"title": str, "explanation": str,
             "supporting_kpis": [{"kpi_name": str, "value": number}],
             "supporting_reviews": [{"review_ref": "R1", "quote": str}]}],
 "improvements": [{"action": str, "cause_index": int, "rationale": str}]}
Give at most 4 causes and 5 improvements. cause_index is the 0-based index of the cause addressed."""


def build_messages(ctx: SellerContext) -> list[Message]:
    payload = {
        "seller_id": ctx.seller_id,
        "kpis": ctx.facts,
        "reviews": [{"ref": ref, "score": r["score"], "text": r["text"]} for ref, r in ctx.reviews.items()],
    }
    user = (
        "KPIs and customer reviews (translated from Portuguese, scores from 1 to 5) for one seller. "
        "Explain the main causes of dissatisfaction and suggest improvements.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=1)
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# 1,234.5 / 229238 / 10.5 ; un nombre collé à une lettre (R3, 2nd) n'est pas extrait
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])")
ALLOWED_FREE_NUMBERS = {1.0, 2.0, 3.0, 4.0, 5.0, 90.0}  # échelle de notes 1-5, « 90 derniers jours »


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".,;:!?\"'").lower()


def numbers_in_text(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in _NUMBER.findall(text)]


def worse_than_marketplace(facts: dict[str, float]) -> list[str]:
    """KPI du vendeur strictement pires que leur référence marketplace."""
    worse = []
    for name, (ref_name, higher_is_worse) in BENCHMARKS.items():
        if name in facts and ref_name in facts:
            seller, ref = facts[name], facts[ref_name]
            if (seller > ref) if higher_is_worse else (seller < ref):
                worse.append(name)
    return worse


def validate_explanation(data: dict, ctx: SellerContext) -> tuple[Explanation | None, list[str]]:
    """Retourne (explication validée par le schéma, liste des problèmes). Liste vide = valide."""
    try:
        expl = Explanation.model_validate(data)
    except ValidationError as exc:
        return None, [f"schéma : {e['loc']} {e['msg']}" for e in exc.errors()]

    issues: list[str] = []
    allowed_values = set(ctx.facts.values()) | ALLOWED_FREE_NUMBERS
    for i, cause in enumerate(expl.causes):
        for kpi in cause.supporting_kpis:
            if kpi.kpi_name not in ctx.facts:
                issues.append(f"cause {i} : KPI inconnu '{kpi.kpi_name}'")
                continue
            if abs(kpi.value - ctx.facts[kpi.kpi_name]) > 1e-6:
                issues.append(f"cause {i} : {kpi.kpi_name} = {kpi.value}, la valeur fournie est {ctx.facts[kpi.kpi_name]}")
            if kpi.kpi_name in NOT_EVIDENCE:
                issues.append(f"cause {i} : {kpi.kpi_name} est une prédiction du modèle, pas une preuve")
            if kpi.kpi_name in BENCHMARKS:
                ref_name, higher_is_worse = BENCHMARKS[kpi.kpi_name]
                ref = ctx.facts.get(ref_name)
                seller = ctx.facts[kpi.kpi_name]
                if ref is not None and (seller <= ref if higher_is_worse else seller >= ref):
                    issues.append(f"cause {i} : {kpi.kpi_name} ({seller}) n'est pas pire que "
                                  f"{ref_name} ({ref}), il ne peut pas appuyer une cause")
        for ref in cause.supporting_reviews:
            review = ctx.reviews.get(ref.review_ref)
            if review is None:
                issues.append(f"cause {i} : avis inexistant {ref.review_ref}")
            elif _normalize(ref.quote) not in _normalize(review["text"]):
                issues.append(f"cause {i} : la citation de {ref.review_ref} n'apparaît pas dans l'avis")
        if not cause.supporting_kpis:
            issues.append(f"cause {i} : aucun KPI cité")
    if not expl.causes:
        # « rien à signaler » n'est acceptable que si aucun KPI n'est effectivement pire
        if worse := worse_than_marketplace(ctx.facts):
            issues.append(f"aucune cause donnée alors que ces KPI sont pires que la marketplace : {worse}")
    for imp in expl.improvements:
        if imp.cause_index >= len(expl.causes):
            issues.append(f"amélioration '{imp.action[:40]}' : cause_index {imp.cause_index} inexistant")

    free_text = [expl.summary] + [c.title + " " + c.explanation for c in expl.causes] + \
                [m.action + " " + m.rationale for m in expl.improvements]
    for text in free_text:
        for n in numbers_in_text(text):
            if not any(abs(n - v) < 1e-6 for v in allowed_values):
                issues.append(f"nombre non fourni dans le texte : {n:g}")
    return expl, sorted(set(issues))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class ExplanationResult:
    seller_id: str
    valid: bool
    explanation: dict | None
    issues: list[str]
    attempts: int
    review_ids: dict[str, str]  # "R1" -> review_id réel, pour tracer les citations


def _parse_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def explain_seller(con: duckdb.DuckDBPyConnection, client: LLMClient, seller_id: str,
                   max_attempts: int = 2) -> ExplanationResult:
    ctx = load_context(con, seller_id)
    review_ids = {ref: r["review_id"] for ref, r in ctx.reviews.items()}
    if not ctx.reviews:
        return ExplanationResult(seller_id, False, None, ["aucun avis avec texte"], 0, review_ids)

    messages = build_messages(ctx)
    issues: list[str] = []
    expl = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.complete(messages)
        except LLMOutputError:
            # JSON invalide rejeté par le fournisseur : essai raté pour CE vendeur seulement.
            # Pas de réponse à renvoyer au modèle, on retente avec les mêmes messages.
            issues = ["la réponse n'est pas un objet JSON"]
            logger.info("%s : essai %d, JSON rejeté par le fournisseur", seller_id[:8], attempt)
            continue
        data = _parse_json(raw)
        if data is None:
            issues = ["la réponse n'est pas un objet JSON"]
        else:
            expl, issues = validate_explanation(data, ctx)
        if not issues:
            return ExplanationResult(seller_id, True, expl.model_dump(), [], attempt, review_ids)
        logger.info("%s : essai %d invalide (%d problèmes)", seller_id[:8], attempt, len(issues))
        # Nouvel essai : on montre au modèle sa réponse et ses erreurs pour qu'il se corrige.
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your answer breaks the rules:\n- " + "\n- ".join(issues)
             + "\nReturn the corrected JSON object only."},
        ]
    return ExplanationResult(seller_id, False, expl.model_dump() if expl else None, issues, max_attempts, review_ids)


def save_results(con: duckdb.DuckDBPyConnection, results: list[ExplanationResult], client: LLMClient) -> None:
    if not results:  # aucun vendeur traité (ex. quota épuisé dès le premier appel)
        return
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold.seller_explanations (
            seller_id VARCHAR, provider VARCHAR, model VARCHAR, prompt_version VARCHAR,
            valid BOOLEAN, attempts INTEGER, explanation JSON, issues JSON, review_ids JSON,
            created_at TIMESTAMP, PRIMARY KEY (seller_id, provider, model, prompt_version)
        )
        """
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.executemany(
        "INSERT OR REPLACE INTO gold.seller_explanations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.seller_id, client.provider, client.model, PROMPT_VERSION, r.valid, r.attempts,
          json.dumps(r.explanation, ensure_ascii=False) if r.explanation else None,
          json.dumps(r.issues, ensure_ascii=False), json.dumps(r.review_ids), now) for r in results],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explications LLM des vendeurs à risque.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--top", type=int, default=5, help="les N vendeurs au plus fort score de risque")
    target.add_argument("--seller", help="un vendeur précis (seller_id)")
    parser.add_argument("--provider", help="mistral ou groq (par défaut : LLM_PROVIDER du .env)")
    args = parser.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.log_level)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        client = make_client(args.provider, settings.cache_dir / LLM_CACHE_FILENAME)
    except LLMError as exc:
        logger.error("%s", exc)
        return 1

    with duckdb.connect(str(settings.duckdb_path)) as con:
        if args.seller:
            sellers = [args.seller]
        else:
            sellers = [s for (s,) in con.execute(
                "SELECT seller_id FROM gold.seller_risk ORDER BY risk_rank LIMIT ?", [args.top]).fetchall()]
        results = []
        for seller_id in sellers:
            try:
                res = explain_seller(con, client, seller_id)
            except LLMError as exc:
                logger.error("%s : %s", seller_id[:8], exc)
                break  # quota épuisé ou clé refusée : inutile d'insister sur les suivants
            results.append(res)
            logger.info("%s : %s en %d essai(s)%s", seller_id[:8], "VALIDE" if res.valid else "INVALIDE",
                        res.attempts, "" if res.valid else f" - {res.issues[:3]}")
        save_results(con, results, client)
    logger.info("%d/%d explications valides | appels API : %d | réponses en cache : %d",
                sum(r.valid for r in results), len(results), client.n_api_calls, client.n_cache_hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
