"""
USA Housing Price Predictor - Streamlit app
Run:  python -m streamlit run app.py

Keep these files in the SAME folder as app.py (or inside a "models" sub-folder):
  *_model.pkl, model_results.csv, USA_Housing.csv
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SEARCH_DIRS = [BASE, BASE / "models", BASE / "model", Path.cwd()]

FEATURES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]

# file stem -> (display name, one-line description)
MODELS = {
    "LinearRegression": ("Linear Regression", "Fits one straight-line relationship per feature."),
    "RidgeRegression": ("Ridge Regression", "Linear model that shrinks large coefficients."),
    "LassoRegression": ("Lasso Regression", "Linear model that can drop weak features."),
    "ElasticNet": ("Elastic Net", "Blend of Ridge and Lasso penalties."),
    "RobustRegression": ("Robust (Huber) Regression", "Linear model less affected by outliers."),
    "PolynomialRegression": ("Polynomial Regression (deg 4)", "Linear model on squared, cubed and higher terms."),
    "SGDRegressor": ("SGD Regressor", "Linear model trained with stochastic gradient descent."),
    "RandomForest": ("Random Forest", "Average of many decision trees."),
    "XGBoost": ("XGBoost", "Gradient-boosted decision trees."),
    "KNN": ("K-Nearest Neighbours", "Averages the most similar neighbourhoods."),
    "SVM": ("Support Vector Regression", "Kernel-based regression."),
    "ANN": ("Neural Network (MLP)", "One hidden layer of 100 neurons."),
}

NAVY = "#007EFD"
AMBER = "#FFF709"
BLUE = "#03D3D3"
TEAL = "#2BF7E3"
CORAL = "#EE0030"
GREY = "#1DB10C"

st.set_page_config(page_title="USA Housing Price Predictor", page_icon="🏠", layout="wide")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Manrope', 'Segoe UI', sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }

/* hero */
.hero {
    background-color: #0B2540;
    background-image:
        linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
    background-size: 28px 28px;
    border-radius: 18px;
    padding: 34px 40px 0 40px;
    display: flex; justify-content: space-between; align-items: flex-end;
    overflow: hidden; margin-bottom: 22px;
}
.hero h1 { color: #fff; font-weight: 800; font-size: 2.2rem; line-height: 1.15;
           margin: 0 0 10px 0; padding: 0; letter-spacing: -0.02em; }
.hero p { color: #C5D3DF; font-size: 1.02rem; max-width: 520px; margin: 0 0 34px 0; line-height: 1.55; }
.hero svg { flex-shrink: 0; margin-left: 24px; }
@media (max-width: 820px) { .hero svg { display: none; } .hero h1 { font-size: 1.7rem; } }

/* result card */
.price-card {
    background: #fff; border-radius: 18px; padding: 28px 30px;
    border-left: 8px solid #F4A93B; box-shadow: 0 2px 14px rgba(11,37,64,.08);
}
.price-card .label { color: #5B6B7A; font-size: .95rem; font-weight: 600; margin-bottom: 4px; }
.price-card .value { color: #0B2540; font-size: 3.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; }
.price-card .range { color: #1B2733; font-size: 1rem; margin-top: 10px; }
.price-card .model { color: #5B6B7A; font-size: .9rem; margin-top: 14px; padding-top: 12px; border-top: 1px solid #E3E9EE; }

.stat {
    background: #fff; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 1px 8px rgba(11,37,64,.06); height: 100%;
}
.stat .k { color: #5B6B7A; font-size: .85rem; font-weight: 600; }
.stat .v { color: #0B2540; font-size: 1.45rem; font-weight: 800; margin-top: 2px; }
.stat .n { color: #5B6B7A; font-size: .8rem; margin-top: 2px; }

.note {
    background: #FFF4DF; border-radius: 12px; padding: 12px 16px;
    color: #5A3F0A; font-size: .93rem; margin: 10px 0;
}
.bad {
    background: #FDE8E6; border-radius: 12px; padding: 12px 16px;
    color: #7A1F16; font-size: .93rem; margin: 10px 0;
}
h2, h3 { color: #0B2540; font-weight: 800; letter-spacing: -0.01em; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #D9E3EB; }
.stTabs [data-baseweb="tab"] { font-weight: 700; padding: 10px 16px; border-radius: 10px 10px 0 0; color: #536779; }
.stTabs [aria-selected="true"] { color: #17324D !important; }

/* Sidebar: calm navy heading, soft neutral background, and clearer controls */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F3F7FA 0%, #FFFFFF 42%);
    border-right: 1px solid #D9E3EB;
}
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
section[data-testid="stSidebar"] h2 {
    color: #17324D; font-size: 1.18rem; font-weight: 800;
    padding: 0 0 10px 0; border-bottom: 2px solid #D9E3EB;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #536779; line-height: 1.45;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] { padding: 5px 0 9px 0; }
section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    border-radius: 10px; border-color: #CBD8E2;
}
section[data-testid="stSidebar"] hr { border-color: #D9E3EB; margin: 1rem 0; }
/* consistent chart cards and readable plot text */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF; border: 1px solid #E2E9EF;
    border-radius: 14px; padding: 8px 8px 0 8px;
    box-shadow: 0 2px 8px rgba(23,50,77,.035);
    margin-bottom: 14px;
}
:focus-visible { outline: 3px solid #E6A23C !important; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

HERO_SVG = """
<svg width="300" height="150" viewBox="0 0 300 150" aria-hidden="true">
  <g fill="#17395B">
    <rect x="6"   y="70" width="46" height="80"/>
    <rect x="62"  y="30" width="56" height="120"/>
    <rect x="128" y="86" width="40" height="64"/>
    <rect x="178" y="52" width="52" height="98"/>
    <rect x="240" y="92" width="52" height="58"/>
  </g>
  <g fill="#F4A93B">
    <rect x="16" y="84" width="8" height="10"/><rect x="34" y="104" width="8" height="10"/>
    <rect x="74" y="46" width="8" height="10"/><rect x="96" y="70" width="8" height="10"/><rect x="74" y="96" width="8" height="10"/>
    <rect x="140" y="102" width="8" height="10"/>
    <rect x="190" y="68" width="8" height="10"/><rect x="210" y="94" width="8" height="10"/>
    <rect x="252" y="106" width="8" height="10"/><rect x="272" y="118" width="8" height="10"/>
  </g>
</svg>
"""

st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>USA Housing Price Predictor</h1>
    <p>Describe a neighbourhood and get an average home price estimate from
    12 machine-learning models you trained, side by side.</p>
  </div>
  {HERO_SVG}
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def find_file(name: str):
    for folder in SEARCH_DIRS:
        path = folder / name
        if path.exists():
            return path
    return None


def money(x: float) -> str:
    return f"${x:,.0f}"


def polish_chart(fig, height=390, top=45):
    """Apply a consistent, low-distraction style to Plotly charts."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=24, t=top, b=18),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Manrope, Segoe UI, sans-serif", color=NAVY, size=12),
        title=dict(font=dict(size=16, color=NAVY), x=0.02, xanchor="left"),
        hoverlabel=dict(bgcolor="#17324D", font_color="#FFFFFF", bordercolor="#17324D"),
        legend=dict(bgcolor="rgba(255,255,255,.88)", bordercolor="#E2E9EF", borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8EEF3", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF3", zeroline=False, automargin=True)
    return fig


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    loaded, errors = {}, {}
    for stem in MODELS:
        path = find_file(f"{stem}_model.pkl")
        if path is None:
            errors[stem] = "file not found"
            continue
        try:
            with open(path, "rb") as f:
                loaded[stem] = pickle.load(f)
        except Exception as exc:  # e.g. xgboost not installed
            errors[stem] = f"{type(exc).__name__}: {exc}"
    return loaded, errors


@st.cache_data
def load_results():
    path = find_file("model_results.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    df["Name"] = df["Model"].map(lambda m: MODELS.get(m, (m, ""))[0])
    return df


@st.cache_data
def load_data():
    path = find_file("USA_Housing.csv")
    return None if path is None else pd.read_csv(path)


models, load_errors = load_models()
results = load_results()
data = load_data()

if not models:
    st.error(
        "No model files were found. Put the *_model.pkl files in the same folder as app.py "
        "(or in a folder named 'models') and refresh."
    )
    st.stop()


def metric_for(stem: str, col: str):
    if results is None:
        return None
    row = results[results["Model"] == stem]
    return None if row.empty else float(row.iloc[0][col])


def is_reliable(stem: str) -> bool:
    r2 = metric_for(stem, "R2 Score")
    return r2 is None or r2 > 0.05  # R2 near/below 0 = no better than guessing the average


def is_strong(stem: str) -> bool:
    r2 = metric_for(stem, "R2 Score")
    return r2 is not None and r2 >= 0.8


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
def feature_range(col: str, default_min: float, default_max: float, default_val: float):
    if data is not None and col in data:
        lo, hi, mid = float(data[col].min()), float(data[col].max()), float(data[col].mean())
        return lo, hi, mid
    return default_min, default_max, default_val


with st.sidebar:
    st.markdown("## 🏡 Neighbourhood details")
    st.caption("Adjust the area averages below. The estimate updates automatically.")
    st.info("These values describe a neighbourhood, not an individual property.", icon="ℹ️")

    lo, hi, mid = feature_range(FEATURES[0], 15000.0, 110000.0, 68000.0)
    income = st.slider("Average area income ($)", float(int(lo)), float(int(hi) + 1), float(int(mid)), 500.0)

    lo, hi, mid = feature_range(FEATURES[1], 2.0, 10.0, 6.0)
    house_age = st.slider("Average house age (years)", round(lo, 1), round(hi + 0.05, 1), round(mid, 1), 0.1)

    lo, hi, mid = feature_range(FEATURES[2], 3.0, 11.0, 7.0)
    rooms = st.slider("Average number of rooms", round(lo, 1), round(hi + 0.05, 1), round(mid, 1), 0.1)

    lo, hi, mid = feature_range(FEATURES[3], 2.0, 6.5, 4.0)
    bedrooms = st.slider("Average number of bedrooms", round(lo, 1), round(hi, 1), round(mid, 1), 0.1)

    lo, hi, mid = feature_range(FEATURES[4], 100.0, 70000.0, 36000.0)
    population = st.slider("Area population", float(int(lo)), float(int(hi) + 1), float(int(mid)), 100.0)

    st.markdown("---")
    ordered = list(models.keys())
    if results is not None:
        ranked = results[results["Model"].isin(ordered)].sort_values("R2 Score", ascending=False)
        ordered = list(ranked["Model"])
    default_idx = 0

    def label_for(stem: str) -> str:
        r2 = metric_for(stem, "R2 Score")
        tag = "" if r2 is None else (f"  (R² {r2:.3f})" if r2 > 0.05 else "  (not usable)")
        return MODELS[stem][0] + tag

    chosen = st.selectbox("Prediction model", ordered, index=default_idx, format_func=label_for)
    st.caption(MODELS[chosen][1])

# Build the input row in the exact column order used for training
X_new = pd.DataFrame([[income, house_age, rooms, bedrooms, population]], columns=FEATURES)


def predict(stem: str) -> float:
    return float(models[stem].predict(X_new)[0])


all_preds = {}
for stem in models:
    try:
        all_preds[stem] = predict(stem)
    except Exception:
        pass

if chosen not in all_preds:
    st.error("The selected model could not make a prediction. Try another model.")
    st.stop()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    ["Prediction", "Model comparison", "Data explorer", "About"]
)

# ----------------------------- Prediction --------------------------------- #
with tab_predict:
    price = all_preds[chosen]
    mae = metric_for(chosen, "MAE")
    r2 = metric_for(chosen, "R2 Score")
    good = is_reliable(chosen)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if good:
            range_html = ""
            if mae is not None:
                range_html = (
                    f'<div class="range">Likely range: <b>{money(max(price - mae, 0))}</b> to '
                    f"<b>{money(price + mae)}</b> (typical error of {money(mae)})</div>"
                )
            st.markdown(
                f"""
<div class="price-card">
  <div class="label">Estimated average home price</div>
  <div class="value">{money(price)}</div>
  {range_html}
  <div class="model">Model: {MODELS[chosen][0]}</div>
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="bad"><b>{MODELS[chosen][0]} gave {price:,.0f}, which is not a usable price.</b><br>
This model failed to learn from the data (R² near or below zero), most likely because the inputs were not scaled.
Choose another model in the sidebar, such as Linear, Lasso or Ridge Regression.</div>""",
                unsafe_allow_html=True,
            )

    with c2:
        strong = [all_preds[s] for s in all_preds if is_strong(s)]
        if strong:
            st.markdown(
                f"""
<div class="stat"><div class="k">Consensus of strong models</div>
<div class="v">{money(float(np.median(strong)))}</div>
<div class="n">Median of {len(strong)} models with R² of 0.80 or higher</div></div>""",
                unsafe_allow_html=True,
            )
        st.write("")
        if r2 is not None and r2 > 0:
            st.markdown(
                f"""
<div class="stat"><div class="k">Accuracy of this model (R²)</div>
<div class="v">{r2:.1%}</div>
<div class="n">Share of price variation it explained on test data</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### How every model sees these inputs")

    rows = []
    for stem, val in all_preds.items():
        if not is_reliable(stem) or not (0 < val < 1e7):
            continue
        tier = "Selected" if stem == chosen else ("R² of 0.80 or higher" if is_strong(stem) else "Lower accuracy")
        rows.append({"Model": MODELS[stem][0], "Price": val, "Group": tier})
    comp = pd.DataFrame(rows).sort_values("Price")

    if not comp.empty:
        fig = px.bar(
            comp,
            x="Price",
            y="Model", 
            orientation="h",
            color="Group",
            color_discrete_map={"Selected": AMBER, "R² of 0.80 or higher": TEAL, "Lower accuracy": GREY},
            text=comp["Price"].map(money),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=max(320, 38 * len(comp) + 90),
            margin=dict(l=0, r=70, t=10, b=0),
            plot_bgcolor="rgba(1,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Predicted price ($)", tickprefix="$", gridcolor="#1278CB"),
            yaxis=dict(title="",color="black", automargin=True),
            legend=dict(title="", orientation="h", y=-0.18),
            font=dict(family="Manrope, Segoe UI, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    hidden = [MODELS[s][0] for s in all_preds if not is_reliable(s) or not (0 < all_preds[s] < 1e7)]
    if hidden:
        st.markdown(
            f'<div class="note">Hidden from the chart because their output is not a usable price: '
            f'{", ".join(hidden)}.</div>',
            unsafe_allow_html=True,
        )

    export = X_new.copy()
    for stem, val in all_preds.items():
        export[f"Predicted price ({MODELS[stem][0]})"] = val
    st.download_button(
        "Download this prediction (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name="house_price_prediction.csv",
        mime="text/csv",
    )

# --------------------------- Model comparison ------------------------------ #
with tab_compare:
    if results is None:
        st.info("Put model_results.csv next to app.py to see the comparison.")
    else:
        res = results.sort_values("R2 Score", ascending=False).reset_index(drop=True)
        best = res.iloc[0]
        st.markdown("### Which model predicts best?")
        st.write(
            f"**{best['Name']}** scored highest on the test set with R² of **{best['R2 Score']:.3f}** "
            f"and an average miss (MAE) of **{money(best['MAE'])}**."
        )

        show = res[["Name", "R2 Score", "MAE", "MSE"]].rename(columns={"Name": "Model"}).copy()
        show["Status"] = np.where(show["R2 Score"] > 0.05, "Usable", "Not usable")
        st.dataframe(
            show,
            hide_index=True,
            column_config={
                "R2 Score": st.column_config.NumberColumn("R²", format="%.4f"),
                "MAE": st.column_config.NumberColumn("MAE ($)", format="%.0f"),
                "MSE": st.column_config.NumberColumn("MSE", format="%.3e"),
            },
        )

        usable = res[res["R2 Score"] > 0.05]
        cc1, cc2 = st.columns(2, gap="large")
        with cc1:
            f1 = px.bar(
                usable.sort_values("R2 Score"),
                x="R2 Score",
                y="Name",
                orientation="h",
                text=usable.sort_values("R2 Score")["R2 Score"].map(lambda v: f"{v:.3f}"),
                title="R² score (higher is better)",
            )
            f1.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
            f1.update_layout(
                height=420, margin=dict(l=0, r=40, t=50, b=0), yaxis_title="", xaxis_title="",
                xaxis_range=[0, 1.08], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope, Segoe UI, sans-serif"),
            )
            st.plotly_chart(f1, use_container_width=True)
        with cc2:
            f2 = px.bar(
                usable.sort_values("MAE", ascending=False),
                x="MAE",
                y="Name",
                orientation="h",
                text=usable.sort_values("MAE", ascending=False)["MAE"].map(money),
                title="Average error in dollars (lower is better)",
            )
            f2.update_traces(marker_color=AMBER, textposition="outside", cliponaxis=False)
            f2.update_layout(
                height=420, margin=dict(l=0, r=70, t=50, b=0), yaxis_title="", xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope, Segoe UI, sans-serif"),
            )
            st.plotly_chart(f2, use_container_width=True)

        st.markdown("### Model performance trade-off")
        tradeoff = usable.dropna(subset=["R2 Score", "MAE"]).copy()
        if not tradeoff.empty:
            tradeoff_fig = px.scatter(
                tradeoff, x="MAE", y="R2 Score", text="Name",
                hover_name="Name", hover_data={"MAE": ":$,.0f", "R2 Score": ":.3f"},
                title="Accuracy vs. average dollar error",
                color="R2 Score", color_continuous_scale=["#B9D8D4", TEAL, NAVY],
                range_color=[max(0, float(tradeoff["R2 Score"].min())), 1],
            )
            tradeoff_fig.update_traces(
                marker=dict(size=13, line=dict(color="#FFFFFF", width=1.4)),
                textposition="top center", textfont=dict(size=10, color=NAVY),
                cliponaxis=False,
            )
            polish_chart(tradeoff_fig, height=460, top=55)
            tradeoff_fig.update_xaxes(title="Mean absolute error ($) — lower is better", tickprefix="$")
            tradeoff_fig.update_yaxes(title="R² score — higher is better", rangemode="tozero")
            tradeoff_fig.update_layout(coloraxis_colorbar=dict(title="R²", thickness=10))
            st.plotly_chart(tradeoff_fig, use_container_width=True)
            st.caption("The most desirable models sit toward the upper-left: higher R² and lower error.")

        bad = res[res["R2 Score"] <= 0.05]["Name"].tolist()
        if bad:
            st.markdown(
                f'<div class="note"><b>{", ".join(bad)}</b> left out of the charts: R² is near or below zero, '
                "so it predicts no better than guessing the average price.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "**What this tells you.** The plain linear models beat the more complex ones here. "
            "SGD, SVM, KNN and the neural network are sensitive to feature scale, and the training "
            "script fed them raw values (income in the tens of thousands next to bedrooms near 4), "
            "which likely explains their weaker scores. See the About tab for the fix."
        )

# ----------------------------- Data explorer ------------------------------- #
with tab_data:
    if data is None:
        st.info("Put USA_Housing.csv next to app.py to explore the data.")
    else:
        st.markdown("### What the training data looks like")
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="stat"><div class="k">Neighbourhoods</div><div class="v">{len(data):,}</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="stat"><div class="k">Average price</div><div class="v">{money(data["Price"].mean())}</div></div>',
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="stat"><div class="k">Price range</div><div class="v">'
            f'{money(data["Price"].min())} to {money(data["Price"].max())}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")

        feature = st.selectbox("Compare price against", FEATURES)
        current = {
            FEATURES[0]: income, FEATURES[1]: house_age, FEATURES[2]: rooms,
            FEATURES[3]: bedrooms, FEATURES[4]: population,
        }[feature]

        sample = data.sample(min(2000, len(data)), random_state=1)
        sc = px.scatter(sample, x=feature, y="Price", opacity=0.62,
                        title=f"Home price vs. {feature}",
                        hover_data={feature: ":,.2f", "Price": ":$,.0f"})
        sc.update_traces(marker=dict(color=BLUE, size=7, line=dict(color="#FFFFFF", width=0.7)),
                         hovertemplate=f"{feature}: %{{x:,.2f}}<br>Price: $%{{y:,.0f}}<extra>Neighbourhood</extra>")
        sc.add_vline(x=current, line_color=AMBER, line_width=3, line_dash="dash",
                     annotation_text="Your input", annotation_position="top right",
                     annotation_font_color=NAVY, annotation_bgcolor="#FFF5E3")
        polish_chart(sc, height=420, top=55)
        sc.update_yaxes(tickprefix="$", title="Home price")
        sc.update_xaxes(title=feature)
        st.plotly_chart(sc, use_container_width=True)

        d1, d2 = st.columns(2, gap="large")
        with d1:
            h = px.histogram(data, x="Price", nbins=35, title="Home price distribution")
            h.update_traces(marker_color=TEAL, marker_line_color="#FFFFFF", marker_line_width=0.7)
            polish_chart(h, height=350, top=52)
            h.update_xaxes(tickprefix="$", title="Home price")
            h.update_yaxes(title="Number of neighbourhoods")
            st.plotly_chart(h, use_container_width=True)
        with d2:
            corr = data[FEATURES + ["Price"]].corr().round(2)
            hm = px.imshow(corr, text_auto=True,
                           color_continuous_scale=["#F3F7FA", "#80B8B0", NAVY],
                           zmin=-1, zmax=1, aspect="auto",
                           title="Feature correlation (−1 to +1)")
            polish_chart(hm, height=390, top=52)
            hm.update_layout(coloraxis_showscale=True, coloraxis_colorbar=dict(title="Correlation", thickness=10))
            hm.update_traces(textfont=dict(size=11, color=NAVY))
            st.plotly_chart(hm, use_container_width=True)

        # Additional views: distribution of a selected feature and price by feature band.
        st.markdown("### Explore a feature in more detail")
        extra_left, extra_right = st.columns(2, gap="large")
        with extra_left:
            feature_hist = px.histogram(
                data, x=feature, nbins=30, title=f"Distribution of {feature}",
            )
            feature_hist.update_traces(marker_color=BLUE, marker_line_color="#FFFFFF", marker_line_width=0.7)
            polish_chart(feature_hist, height=350, top=52)
            feature_hist.update_xaxes(title=feature)
            feature_hist.update_yaxes(title="Number of neighbourhoods")
            st.plotly_chart(feature_hist, use_container_width=True)
        with extra_right:
            band_data = data[[feature, "Price"]].dropna().copy()
            try:
                band_data["Feature band"] = pd.qcut(band_data[feature], q=4, duplicates="drop")
                band_data["Feature band"] = band_data["Feature band"].astype(str)
                band_order = band_data["Feature band"].drop_duplicates().tolist()
                band_plot = px.box(
                    band_data, x="Feature band", y="Price", points=False,
                    category_orders={"Feature band": band_order},
                    title=f"Price across {feature} quartiles",
                    color_discrete_sequence=[TEAL],
                )
                band_plot.update_traces(line=dict(color=TEAL, width=2), fillcolor="rgba(42,140,130,.22)")
                polish_chart(band_plot, height=350, top=52)
                band_plot.update_yaxes(title="Home price", tickprefix="$", tickformat="~s")
                band_plot.update_xaxes(title="Low → high feature range", tickangle=-15)
                st.plotly_chart(band_plot, use_container_width=True)
            except (ValueError, TypeError):
                st.caption("Quartile view is unavailable because this feature has too few distinct values.")

        with st.expander("Browse the raw data"):
            st.dataframe(data.drop(columns=["Address"], errors="ignore").head(200), hide_index=True, use_container_width=True)

# --------------------------------- About ----------------------------------- #
with tab_about:
    st.markdown("### About this project")
    st.write(
        "This app loads the 12 regression models saved by `GAURAV.PY`, trained on the "
        "USA Housing dataset (5 neighbourhood features predicting average home price). "
        "Everything runs locally on your computer."
    )

    st.markdown("#### Run it")
    st.code(
        "pip install -r requirements.txt\npython -m streamlit run app.py",
        language="bash",
    )

    st.markdown("#### Good to know")
    st.markdown(
        "- Inputs are **area averages**, so the result is the typical price in that neighbourhood, not a quote for one house.\n"
        "- The models were trained with scikit-learn on your machine. If you see a version warning, "
        "install the same scikit-learn version you trained with.\n"
        "- Models were trained on **unscaled** features. That is why SGD diverged and SVM, KNN and the "
        "neural network score low."
    )

    st.markdown("#### Improve the training script")
    st.write("Wrap scale-sensitive models in a pipeline with a scaler, then retrain and replace the .pkl files:")
    st.code(
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "'SGDRegressor': Pipeline([('scale', StandardScaler()), ('model', SGDRegressor())]),\n"
        "'SVM':          Pipeline([('scale', StandardScaler()), ('model', SVR(C=100000))]),\n"
        "'KNN':          Pipeline([('scale', StandardScaler()), ('model', KNeighborsRegressor())]),\n"
        "'ANN':          Pipeline([('scale', StandardScaler()), ('model', MLPRegressor(max_iter=1000))]),",
        language="python",
    )
    st.caption("This app needs no changes after retraining, because it only calls model.predict().")

    if load_errors:
        with st.expander("Models that did not load"):
            for stem, err in load_errors.items():
                st.write(f"**{MODELS[stem][0]}**: {err}")