"""Interface Streamlit : vendeurs à risque, KPI, évolution mensuelle, explications LLM.

Lancement : streamlit run app.py
Affichage seulement : les requêtes sont dans olist_insights.app_data (testées à part).
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from olist_insights import app_data
from olist_insights.config import load_settings
from olist_insights.explain import PROMPT_VERSION

st.set_page_config(page_title="Olist · vendeurs à risque", page_icon="📦", layout="wide")

DB_PATH = load_settings().duckdb_path


# --- Chargement (mis en cache : une requête par vendeur, pas à chaque clic) ------------------

@st.cache_data(show_spinner=False)
def load_ranking() -> pd.DataFrame:
    with app_data.connect(DB_PATH) as con:
        return app_data.risk_ranking(con)


@st.cache_data(show_spinner=False)
def load_seller(seller_id: str) -> tuple[dict, pd.DataFrame, dict | None, dict]:
    with app_data.connect(DB_PATH) as con:
        facts = app_data.seller_facts(con, seller_id)
        monthly = app_data.seller_monthly(con, seller_id)
        expl = app_data.seller_explanation(con, seller_id)
        reviews = app_data.cited_reviews(con, seller_id, expl["review_ids"]) if expl else {}
    return facts, monthly, expl, reviews


if not DB_PATH.is_file():
    st.error(f"Base DuckDB introuvable : `{DB_PATH}`. Lancer d'abord le pipeline (voir le README).")
    st.stop()

try:
    ranking = load_ranking()
except Exception as exc:  # base verrouillée par un pipeline en cours, table gold manquante…
    st.error(f"Lecture de la base impossible : {exc}")
    st.info("Si un script du pipeline tourne (il verrouille la base), relancer l'app quand il a fini. "
            "Si gold.seller_risk manque : `python -m olist_insights.model`.")
    st.stop()

summary = app_data.model_summary()

# --- En-tête ---------------------------------------------------------------------------------

st.title("📦 Olist · vendeurs dont la satisfaction risque de chuter")
st.caption("Données publiques Olist (Brésil, 2016-2018). Un modèle classe les vendeurs selon le risque que "
           "≥ 25 % de leurs avis des 90 prochains jours soient à 1-2 étoiles ; un LLM explique les causes "
           "à partir des avis clients traduits.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vendeurs notés", f"{len(ranking):,}".replace(",", " "))
c2.metric("Date du score", "sept. 2018")
c3.metric("Explications LLM validées", int(ranking["has_explanation"].sum()))
if summary:
    c4.metric("Précision du top 10 % (test)", f"{summary['model']['precision_top10']:.0%}",
              help=f"Part de vrais cas parmi les 10 % de vendeurs jugés les plus risqués, sur le jeu de test. "
                   f"Taux de base : {summary['prevalence']:.0%}.")

tab_rank, tab_seller, tab_method = st.tabs(["Classement du risque", "Fiche vendeur", "Méthode et limites"])

# --- Onglet 1 : classement -------------------------------------------------------------------

with tab_rank:
    f1, f2, f3 = st.columns([2, 2, 1])
    states = f1.multiselect("État", sorted(ranking["state"].dropna().unique()))
    cats = f2.multiselect("Catégorie principale", sorted(ranking["main_category"].dropna().unique()))
    only_expl = f3.checkbox("Avec explication", help="Seuls les vendeurs ayant une explication LLM validée")

    view = ranking
    if states:
        view = view[view["state"].isin(states)]
    if cats:
        view = view[view["main_category"].isin(cats)]
    if only_expl:
        view = view[view["has_explanation"]]

    shown = view.assign(
        risk_pct=view["risk_score"] * 100,
        low_pct=view["low_rate_90d"] * 100,
        late_pct=view["late_rate_90d"] * 100,
        handoff_pct=view["late_handoff_rate_90d"] * 100,
    )[["risk_rank", "seller_id", "risk_pct", "state", "main_category", "n_reviews_90d",
       "avg_rating_90d", "low_pct", "late_pct", "handoff_pct", "has_explanation"]]

    st.dataframe(
        shown, hide_index=True, width="stretch", height=480,
        column_config={
            "risk_rank": st.column_config.NumberColumn("Rang"),
            "seller_id": st.column_config.TextColumn("Vendeur"),
            "risk_pct": st.column_config.ProgressColumn("Score de risque", format="%.0f", min_value=0, max_value=100),
            "state": "État",
            "main_category": "Catégorie",
            "n_reviews_90d": st.column_config.NumberColumn("Avis 90 j"),
            "avg_rating_90d": st.column_config.NumberColumn("Note 90 j", format="%.2f"),
            "low_pct": st.column_config.NumberColumn("1-2 ★ 90 j (%)", format="%.0f"),
            "late_pct": st.column_config.NumberColumn("Retards 90 j (%)", format="%.0f"),
            "handoff_pct": st.column_config.NumberColumn("Remise tardive 90 j (%)", format="%.0f"),
            "has_explanation": st.column_config.CheckboxColumn("Explication"),
        },
    )
    st.caption(f"{len(view)} vendeurs affichés. Le score sert à **trier** les vendeurs à examiner, "
               "pas à prouver un problème : voir l'onglet « Méthode et limites ».")

# --- Onglet 2 : fiche vendeur ----------------------------------------------------------------

with tab_seller:
    def label(row) -> str:
        mark = " · ✅ explication" if row.has_explanation else ""
        return f"#{row.risk_rank} · {row.seller_id[:8]} · {row.main_category} ({row.state}){mark}"

    options = ranking.sort_values(["has_explanation", "risk_rank"], ascending=[False, True])
    # libellé -> seller_id (les libellés sont uniques : ils contiennent le rang)
    by_label = {label(row): row.seller_id for row in options.itertuples()}
    seller_id = by_label[st.selectbox("Vendeur (les vendeurs avec explication en premier)", list(by_label))]

    facts, monthly, expl, reviews = load_seller(seller_id)
    row = ranking.set_index("seller_id").loc[seller_id]

    st.subheader(f"Vendeur {seller_id}")
    st.caption(f"Catégorie principale : {row['main_category']} · État : {row['state']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rang de risque", f"#{row['risk_rank']} / {len(ranking)}")
    m2.metric("Commandes (total)", facts.get("n_orders"))
    m3.metric("Note moyenne", facts.get("avg_rating"),
              delta=round(facts["avg_rating"] - facts["marketplace_avg_rating"], 2) if "avg_rating" in facts else None,
              help="Écart avec la moyenne de la marketplace")

    with st.container():
        st.markdown("**KPI comparés à la marketplace**")
        comp = app_data.kpi_comparison(facts)
        st.dataframe(
            comp.style.apply(lambda r: ["background-color: rgba(220, 60, 60, .15)" if r["pire_que_marketplace"]
                                        else "" for _ in r], axis=1).format({"vendeur": "{:g}", "marketplace": "{:g}"}),
            hide_index=True, width="stretch",
            column_config={"kpi": "KPI", "vendeur": "Vendeur", "marketplace": "Marketplace",
                           "pire_que_marketplace": st.column_config.CheckboxColumn("Pire")},
        )
    with st.container():
        st.markdown("**Évolution mensuelle**")
        base = alt.Chart(monthly).encode(x=alt.X("yearmonth(month):T", title=None))
        orders = base.mark_bar(opacity=0.35).encode(
            y=alt.Y("n_orders:Q", title="Commandes"), tooltip=["yearmonth(month):T", "n_orders:Q"])
        # taille du point = nombre d'avis du mois : une note tirée d'un seul avis se voit petite
        line = base.mark_line(color="#d9534f").encode(
            y=alt.Y("avg_rating:Q", title="Note moyenne", scale=alt.Scale(domain=[1, 5])))
        points = base.mark_circle(color="#d9534f", opacity=0.9).encode(
            size=alt.Size("n_reviews:Q", title="Avis du mois", scale=alt.Scale(range=[15, 250])),
            y=alt.Y("avg_rating:Q", title="Note moyenne", scale=alt.Scale(domain=[1, 5])),
            tooltip=["yearmonth(month):T", alt.Tooltip("avg_rating:Q", format=".2f"), "n_reviews:Q"])
        st.altair_chart(alt.layer(orders, line + points).resolve_scale(y="independent"), width="stretch")
        st.caption("Barres : commandes par mois d'achat. Ligne : note moyenne par mois de l'avis ; "
                   "plus le point est petit, moins il y a d'avis ce mois-là (une note sur 1 ou 2 avis est peu fiable).")

    st.divider()
    st.markdown("### Explication des causes (LLM)")
    if expl is None:
        st.info("Pas d'explication validée pour ce vendeur (seul le top 10 du risque a été traité, pour "
                "rester dans le quota gratuit). Pour la générer :\n\n"
                f"`python -m olist_insights.explain --seller {seller_id}`")
    else:
        e = expl["explanation"]
        st.markdown(e["summary"])
        if not e["causes"]:
            st.success("Le LLM ne trouve aucun KPI pire que la marketplace : le score de risque "
                       "n'est pas confirmé par les données (possible faux positif du modèle).")
        for i, cause in enumerate(e["causes"]):
            with st.expander(f"Cause {i + 1} · {cause['title']}", expanded=i == 0):
                st.markdown(cause["explanation"])
                st.markdown("**KPI cités** : " + ", ".join(
                    f"`{k['kpi_name']}` = {k['value']:g}" for k in cause["supporting_kpis"]))
                for ref in cause["supporting_reviews"]:
                    review = reviews.get(ref["review_ref"])
                    stars = f" · {review['score']}★" if review else ""
                    st.markdown(f"> « {ref['quote']} » — *{ref['review_ref']}{stars}*")
                    if review:
                        with st.popover(f"Avis complet {ref['review_ref']}"):
                            st.markdown(f"**Traduction** : {review['en']}")
                            st.markdown(f"**Original (portugais)** : {review['pt']}")
                fixes = [imp for imp in e["improvements"] if imp["cause_index"] == i]
                if fixes:
                    st.markdown("**Améliorations proposées**")
                    for imp in fixes:
                        st.markdown(f"- **{imp['action']}** — {imp['rationale']}")
        st.caption(f"Généré par `{expl['model']}` via {expl['provider']} (prompt {expl['prompt_version']}, "
                   f"{expl['attempts']} essai(s)). Validé automatiquement : chaque chiffre est un KPI fourni, "
                   "chaque citation est retrouvée mot pour mot dans l'avis, chaque KPI cité est pire que la "
                   "marketplace. Les avis sont traduits automatiquement (MarianMT).")

# --- Onglet 3 : méthode et limites ------------------------------------------------------------

with tab_method:
    st.markdown("#### Ce que prédit le score")
    if summary:
        t = summary["target"]
        st.markdown(
            f"Probabilité qu'au moins **{t['bad_threshold']:.0%}** des avis des **{t['horizon_days']} prochains jours** "
            f"soient à 1-2 étoiles (vendeurs avec ≥ {t['min_past_reviews']} avis sur les {t['lookback_days']} "
            "derniers jours). Calculé uniquement avec des données antérieures à la date du score.")
        perf = pd.DataFrame([
            {"Méthode": "Hasard (taux de base)", **summary["constant"]},
            {"Méthode": "Règle simple : « déjà beaucoup de mauvaises notes »", **summary["rule"]},
            {"Méthode": f"Modèle retenu ({summary['selected_model']})", **summary["model"]},
        ])
        st.dataframe(perf, hide_index=True, width="stretch", column_config={
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
            "precision_top10": st.column_config.NumberColumn("Précision top 10 %", format="%.3f"),
            "pr_auc_previously_ok": st.column_config.NumberColumn("PR-AUC, vendeurs jusque-là sains", format="%.3f"),
        })
        base_ok = summary["constant"]["pr_auc_previously_ok"]
        lift_model = summary["model"]["pr_auc_previously_ok"] / base_ok
        lift_rule = summary["rule"]["pr_auc_previously_ok"] / base_ok
        st.warning(
            "**Limite honnête** : sur le jeu de test, la règle simple fait mieux que le modèle sur l'ensemble des "
            "vendeurs. Le modèle n'apporte un gain que pour repérer les vendeurs **jusque-là sains** qui vont "
            f"décrocher (PR-AUC ×{lift_model:.1f} par rapport au hasard, contre ×{lift_rule:.1f} pour la règle), "
            "c'est-à-dire les cas qu'on ne voit pas déjà. Le score est un outil de tri, pas une preuve.")
    scores = app_data.eval_summary()
    if scores:
        st.markdown("#### Qualité de la traduction et des explications")
        tr, ca = scores.get("translation") or {}, scores.get("llm_causes") or {}
        k1, k2, k3 = st.columns(3)
        if tr:
            k1.metric("Avis traduits avec le sens conservé", f"{tr['meaning_preserved_pct']:.0f} %",
                      help=f"Sur {tr['n_graded']} avis relus ; sens perdu : {tr['meaning_lost_pct']:.1f} %")
        if ca:
            k2.metric("Causes appuyées par les avis cités", f"{ca['supported_by_reviews_pct']:.0f} %")
            k3.metric("Vraies causes (pas des symptômes)", f"{ca['is_cause_not_symptom_pct']:.0f} %",
                      help=f"Prompt {PROMPT_VERSION} ; échantillon de {ca['is_cause_not_symptom_n']} causes. "
                           "En partie garanti par construction : depuis la v4, un contrôle automatique rejette "
                           "les titres qui ne parlent que de la note.")
        st.caption(f"Relecture : {scores.get('grader', 'non précisée')}. Petits échantillons : ordres de grandeur, "
                   "pas des mesures précises.")
