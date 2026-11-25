import copy
import json

import duckdb
import pytest

from olist_insights import explain
from olist_insights.config import load_settings
from olist_insights.explain import (
    SellerContext,
    explain_seller,
    is_symptom_title,
    numbers_in_text,
    validate_explanation,
)
from olist_insights.llm_client import LLMOutputError

CTX = SellerContext(
    seller_id="s1",
    facts={"late_delivery_rate_pct": 31.2, "avg_rating": 3.5, "n_orders": 120,
           "marketplace_late_delivery_rate_pct": 6.7, "marketplace_avg_rating": 3.0,
           "risk_score_pct": 90.0},
    reviews={
        "R1": {"review_id": "rid-1", "score": 1, "text": "The product arrived two weeks late. Terrible."},
        "R2": {"review_id": "rid-2", "score": 2, "text": "I never received the second item."},
        "R3": {"review_id": "rid-3", "score": 5, "text": "Great product, fast delivery."},
    },
)

VALID = {
    "summary": "Late deliveries (31.2% versus 6.7% on the marketplace) drive low ratings.",
    "causes": [{
        "title": "Late deliveries",
        "explanation": "Customers complain about delays; the late rate is far above the marketplace.",
        "supporting_kpis": [{"kpi_name": "late_delivery_rate_pct", "value": 31.2},
                            {"kpi_name": "marketplace_late_delivery_rate_pct", "value": 6.7}],
        "supporting_reviews": [{"review_ref": "R1", "quote": "arrived two weeks late"}],
    }],
    "improvements": [{"action": "Hand parcels to the carrier earlier", "cause_index": 0,
                      "rationale": "Reduces the delivery delay."}],
}


def _with(**changes):
    data = copy.deepcopy(VALID)
    for path, value in changes.items():
        target = data
        *keys, last = path.split("__")
        for k in keys:
            target = target[int(k)] if k.isdigit() else target[k]
        target[int(last) if last.isdigit() else last] = value
    return data


def test_valid_explanation_passes():
    expl, issues = validate_explanation(VALID, CTX)
    assert issues == [] and expl.causes[0].title == "Late deliveries"


@pytest.mark.parametrize("data, expected", [
    (_with(summary="Late rate is 45% and growing."), "nombre non fourni"),               # chiffre inventé
    (_with(summary="Late rate is 31% here."), "nombre non fourni"),                      # arrondi interdit
    (_with(causes__0__supporting_kpis__0__value=30.0), "la valeur fournie est 31.2"),    # valeur modifiée
    (_with(causes__0__supporting_kpis__0__kpi_name="refund_rate"), "KPI inconnu"),       # KPI inventé
    (_with(causes__0__supporting_reviews__0__review_ref="R9"), "avis inexistant R9"),    # avis inventé
    (_with(causes__0__supporting_reviews__0__quote="arrived broken"), "n'apparaît pas"), # citation inventée
    (_with(causes__0__supporting_kpis=[]), "aucun KPI"),
    (_with(improvements__0__cause_index=3), "cause_index 3"),
    (_with(causes__0__supporting_reviews=[]), "schéma"),                                 # cause sans avis
    # chiffre exact mais conclusion fausse : la note 3.5 n'est pas pire qu'une marketplace à 3.0
    (_with(causes__0__supporting_kpis=[{"kpi_name": "avg_rating", "value": 3.5}]), "n'est pas pire"),
    (_with(causes__0__supporting_kpis=[{"kpi_name": "risk_score_pct", "value": 90.0}]), "pas une preuve"),
    (_with(causes__0__title="High proportion of low ratings"), "le symptôme"),        # symptôme (v4)
])
def test_invalid_explanations_are_caught(data, expected):
    _, issues = validate_explanation(data, CTX)
    assert any(expected in issue for issue in issues), issues


NOTHING_TO_REPORT = {"summary": "The data does not confirm a satisfaction problem.", "causes": [], "improvements": []}


def test_no_cause_rejected_when_a_kpi_is_worse():
    # CTX : retard de 31.2 % contre 6.7 % sur la marketplace -> « rien à signaler » est faux
    _, issues = validate_explanation(NOTHING_TO_REPORT, CTX)
    assert any("late_delivery_rate_pct" in issue for issue in issues)


def test_no_cause_accepted_when_seller_beats_marketplace():
    good = SellerContext("s2", {"late_delivery_rate_pct": 2.0, "marketplace_late_delivery_rate_pct": 6.7}, CTX.reviews)
    assert validate_explanation(NOTHING_TO_REPORT, good)[1] == []


@pytest.mark.parametrize("title, expected", [
    # titres relevés à l'étape 6 (notés « symptôme »)
    ("High proportion of low ratings", True),
    ("Below‑average overall rating", True),   # tiret insécable produit par le LLM
    ("High low‑rating rate", True),
    # formulations jamais vues, pour ne pas juger le détecteur sur ses seuls exemples d'origine
    ("Poor customer review scores", True),
    ("Many negative reviews", True),
    # vraies causes, même quand elles parlent aussi de la note
    ("Product authenticity and mismatch problems driving low ratings", False),
    ("Damaged products", False),
    ("No answer from the seller", False),
    ("High late delivery rate", False),
])
def test_is_symptom_title(title, expected):
    assert is_symptom_title(title) is expected


def test_quote_matching_ignores_case_and_spacing():
    data = _with(causes__0__supporting_reviews__0__quote="ARRIVED  two weeks LATE.")
    assert validate_explanation(data, CTX)[1] == []


def test_numbers_in_text():
    # R3 (collé à une lettre) est ignoré ; le 5 de "5-star" est extrait mais autorisé (échelle 1-5)
    assert numbers_in_text("31.2% of 1,200 orders in 90 days, see R3 and 5-star") == [31.2, 1200.0, 90.0, 5.0]


class FakeLLM:
    """Faux LLM : renvoie les réponses prévues ; une exception dans la liste est levée."""
    provider, model = "fake", "fake-model"

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, messages, json_mode=True):
        self.calls.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setattr(explain, "load_context", lambda con, seller_id: CTX)


def test_explain_seller_first_try(ctx):
    llm = FakeLLM([json.dumps(VALID)])
    res = explain_seller(None, llm, "s1")
    assert res.valid and res.attempts == 1 and res.review_ids["R1"] == "rid-1"


def test_explain_seller_retries_with_feedback(ctx):
    bad = _with(summary="Late rate is 45%.")
    llm = FakeLLM([json.dumps(bad), json.dumps(VALID)])
    res = explain_seller(None, llm, "s1")
    assert res.valid and res.attempts == 2
    feedback = llm.calls[1][-1]["content"]
    assert "nombre non fourni dans le texte : 45" in feedback  # le modèle voit ses erreurs


def test_explain_seller_marked_invalid_after_retries(ctx):
    llm = FakeLLM(["not json", "still not json"])
    res = explain_seller(None, llm, "s1")
    assert not res.valid and res.issues == ["la réponse n'est pas un objet JSON"]


def test_provider_json_rejection_is_one_failed_attempt(ctx):
    llm = FakeLLM([LLMOutputError("JSON invalide"), json.dumps(VALID)])
    res = explain_seller(None, llm, "s1")
    assert res.valid and res.attempts == 2


def test_save_results_with_nothing_to_save():
    with duckdb.connect(":memory:") as con:
        explain.save_results(con, [], FakeLLM([]))  # ne doit pas planter


def test_prompt_contains_only_provided_facts():
    messages = explain.build_messages(CTX)
    payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
    assert payload["kpis"] == CTX.facts
    assert [r["ref"] for r in payload["reviews"]] == ["R1", "R2", "R3"]
    assert "rid-1" not in messages[1]["content"]  # les vrais identifiants ne sont pas envoyés


# --- Intégration : contexte construit depuis la vraie base (ignoré si absente) ---
DB = load_settings().duckdb_path


@pytest.mark.skipif(not DB.is_file(), reason="base DuckDB absente")
def test_real_context_for_riskiest_seller():
    with duckdb.connect(str(DB), read_only=True) as con:
        (seller,) = con.execute("SELECT seller_id FROM gold.seller_risk ORDER BY risk_rank LIMIT 1").fetchone()
        ctx = explain.load_context(con, seller)
    assert "late_delivery_rate_pct" in ctx.facts and "risk_score_pct" in ctx.facts
    assert 0 < len(ctx.reviews) <= explain.MAX_NEGATIVE_REVIEWS + explain.MAX_OTHER_REVIEWS
    assert all(r["text"] for r in ctx.reviews.values())
