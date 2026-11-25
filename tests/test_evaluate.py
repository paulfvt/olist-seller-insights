import json

import duckdb
import pytest

from olist_insights import evaluate
from olist_insights.evaluate import (
    length_ratio_flag,
    portuguese_leftovers,
    reviews_from_ids,
    summarize_cause_grades,
    summarize_translation_grades,
)
from olist_insights.translate import (
    has_repetition_loop,
    is_probable_omission,
    is_untreated_caps,
    join_pieces,
    split_sentences,
)


def test_split_sentences():
    assert split_sentences("Chegou rápido. Muito bom!  Recomendo\nObrigado") == \
        ["Chegou rápido.", "Muito bom!", "Recomendo", "Obrigado"]
    assert split_sentences("Produto ok") == ["Produto ok"]
    assert split_sentences("Comprei 10.5 metros") == ["Comprei 10.5 metros"]  # pas de coupe dans un nombre


def test_omission_detected_only_for_multi_sentence_texts():
    pt = "Prazo de entrega não foi cumprido. Encomenda ainda não chegou."
    assert is_probable_omission(pt, "Delivery time hasn't been met.")
    assert not is_probable_omission(pt, "Delivery time has not been met. Order's not here yet.")
    # une seule phrase : une traduction plus courte n'est pas une phrase perdue
    assert not is_probable_omission("Produto chegou em perfeitas condições e no prazo", "Fine.")
    # phrase courte perdue (vue à la relecture) : longueur correcte mais une phrase en moins
    pt2 = "Está apontando como entregue no rastreamento, mas o produto não foi recebido. Ainda estou no aguardo."
    assert is_probable_omission(pt2, "It shows as delivered in the tracking, but the product wasn't received.")


def test_join_pieces_keeps_sentence_count():
    src = "produto bom\nótimo site, eu recomendo"
    joined = join_pieces(["good product", "great site, I recommend"])
    assert joined == "good product. great site, I recommend"
    assert not is_probable_omission(src, joined)  # plus de faux positif à chaque passage
    assert join_pieces(["Fast!", "Thanks"]) == "Fast! Thanks"  # ponctuation existante gardée


def test_untreated_caps_selection_is_idempotent():
    src = "NA MINHA OPINIÃO RECOMENDO O MEU PRODUTO FOI ENTREGUE ANTES DO PRASO"
    assert is_untreated_caps(src, "IN MY OPINION I RECOMMENDED MY PRODUCT WAS DELIVERED BEFORE THE PRACE")
    # une fois retraduit en casse de phrase : plus sélectionné (pas de retraduction sans fin)
    assert not is_untreated_caps(src, "In my opinion I recommend my product was delivered before the deadline")
    assert not is_untreated_caps("Produto OK, chegou rápido", "Product OK, arrived fast")  # sigle, pas un cri
    # nom de vendeur anonymisé en minuscules au milieu d'un texte en majuscules
    assert is_untreated_caps("GOSTO DE COMPRAR NA stark,", "I LIKE TO BUY IN THE STARK,.")


def test_normalize_case_handles_anonymised_names():
    from olist_insights.translate import normalize_case
    assert normalize_case("GOSTO DE COMPRAR NA stark") == "Gosto de comprar na stark"
    assert normalize_case("Produto OK") == "Produto OK"


def test_punctuation_loop_is_a_loop():
    assert has_repetition_loop("Delivery before deadline . . . . . . . . . .")
    assert not has_repetition_loop("Fast delivery. Good product. Recommend.")


def test_length_ratio_flag():
    pt = "Prazo de entrega não foi cumprido. Encomenda ainda não chegou."
    assert length_ratio_flag(pt, "Delivery time hasn't been met.") == "omission"
    assert length_ratio_flag("Muito bom produto, recomendo a todos", "I " * 60) == "long"
    assert length_ratio_flag("ok", "Okay, fine.") is None  # trop court pour juger


def test_portuguese_leftovers():
    assert portuguese_leftovers("GOSTEI, IT'S OF QUALITY.") == {"gostei"}
    assert portuguese_leftovers("Very good product, fast delivery.") == set()


def test_reviews_from_ids_checks_seller_ownership():
    with duckdb.connect(":memory:") as con:
        con.execute("CREATE SCHEMA silver")
        con.execute("CREATE TABLE silver.seller_reviews AS SELECT * FROM (VALUES "
                    "('rA', 's1'), ('rB', 's2')) t(review_id, seller_id)")
        con.execute("CREATE TABLE silver.review_translations AS SELECT * FROM (VALUES "
                    "('rA', 'Late again.'), ('rB', 'Great.')) t(review_id, review_en)")
        got = reviews_from_ids(con, "s1", {"R1": "rA", "R2": "rB"})
    # rB appartient à un autre vendeur : une citation attribuée à s1 serait fausse
    assert got == {"R1": {"review_id": "rA", "text": "Late again."}}


def test_summarize_translation_grades():
    rows = [{"score": 1, "adequacy_0_2": "2", "fluency_0_2": "1"},
            {"score": 1, "adequacy_0_2": "0", "fluency_0_2": "0"},
            {"score": 5, "adequacy_0_2": "2", "fluency_0_2": "2"},
            {"score": 5, "adequacy_0_2": "", "fluency_0_2": ""}]  # pas encore noté : ignoré
    s = summarize_translation_grades(rows)
    assert s["n_graded"] == 3
    assert s["adequacy_mean_0_2"] == pytest.approx(4 / 3, abs=0.01)
    assert s["meaning_lost_pct"] == pytest.approx(33.3)
    assert s["adequacy_mean_by_review_score"] == {"1": 1.0, "5": 2.0}
    assert summarize_translation_grades([{"score": 1, "adequacy_0_2": ""}]) is None


def test_summarize_cause_grades():
    rows = [{"supported_by_reviews": "y", "is_cause_not_symptom": "n", "actionable": "y"},
            {"supported_by_reviews": "Y", "is_cause_not_symptom": "y", "actionable": ""}]
    s = summarize_cause_grades(rows)
    assert s["supported_by_reviews_pct"] == 100.0 and s["is_cause_not_symptom_pct"] == 50.0
    assert s["actionable_n"] == 1  # réponse vide ignorée


@pytest.mark.parametrize("delimiter", [";", ","])
def test_csv_roundtrip_any_delimiter(tmp_path, delimiter):
    rows = [{"review_id": "r1", "review_pt": "Não; chegou, ainda", "adequacy_0_2": "2"}]
    path = tmp_path / "grid.csv"
    if delimiter == ";":
        evaluate._write_csv(path, rows)  # format écrit par le script (Excel FR)
    else:  # même grille ré-enregistrée par un autre outil, avec des virgules
        import csv
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    assert evaluate._read_csv(path) == rows


def test_issue_types_are_grouped():
    assert evaluate._issue_type("nombre non fourni dans le texte : 45") == "invented_number"
    assert evaluate._issue_type("cause 0 : la citation de R2 n'apparaît pas dans l'avis") == "quote_not_in_review"


# --- Intégration : évaluation complète sur la vraie base (ignorée si absente) ---
DB = evaluate.load_settings().duckdb_path


@pytest.mark.skipif(not DB.is_file(), reason="base DuckDB absente")
def test_real_translation_checks_and_sample():
    with duckdb.connect(str(DB), read_only=True) as con:
        checks = evaluate.translation_checks(con)
        sample = evaluate.translation_sample(con)
    assert checks["n_missing"] == 0
    assert len(sample) == 30 and sorted({r["score"] for r in sample}) == [1, 2, 3, 4, 5]
    json.dumps(checks)  # le rapport doit être sérialisable
