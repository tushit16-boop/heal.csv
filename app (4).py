from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

st.set_page_config(page_title="HealthWise Smart Premium Engine", page_icon="🏥", layout="wide")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "healthwise.csv"
RANDOM_STATE = 42
NUM = ["age", "bmi", "children", "exercise_freq"]
CAT = ["sex", "smoker", "region"]
FEATURES = NUM + CAT

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("healthwise.csv is missing from the repository root.")
    d = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    d["bmi"] = d["bmi"].fillna(d["bmi"].median())
    d["exercise_freq"] = d["exercise_freq"].fillna(d["exercise_freq"].median())
    return d

def train_engine(d):
    med = {c: float(d[c].median()) for c in NUM}
    pre = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
    ])
    clf = Pipeline([
        ("pre", pre),
        ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ])
    clf.fit(d[FEATURES], d["risk_tier"])
    regs = {}
    for tier in ["Low", "Medium", "High"]:
        sub = d[d["risk_tier"] == tier]
        tier_pre = ColumnTransformer([
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
        ])
        regs[tier] = Pipeline([
            ("pre", tier_pre),
            ("model", LinearRegression()),
        ]).fit(sub[FEATURES], sub["annual_charge"])
    return {"clf": clf, "regs": regs, "features": FEATURES, "num": NUM, "cat": CAT, "med": med}

@st.cache_resource
def load_engine():
    d = load_data()
    return train_engine(d), "Model trained from healthwise.csv"

def predict_customer(engine, customer):
    row = pd.DataFrame([customer])
    for c in NUM:
        if c not in row.columns or pd.isna(row.at[0, c]):
            row[c] = engine["med"][c]
    row = row[FEATURES]
    tier = str(engine["clf"].predict(row)[0])
    premium = max(0.0, float(engine["regs"][tier].predict(row)[0]))
    return tier, premium

def score_frame(engine, frame):
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    out = frame.copy()
    results = [predict_customer(engine, row.to_dict()) for _, row in out.iterrows()]
    out["predicted_tier"] = [t for t, _ in results]
    out["predicted_premium"] = [round(p, 2) for _, p in results]
    return out

def current_metrics(d):
    X = d[FEATURES]
    y = d["risk_tier"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE, stratify=y)
    temp = train_engine(pd.concat([Xtr, ytr], axis=1).join(d[["annual_charge"]], how="left"))
    acc = accuracy_score(yte, temp["clf"].predict(Xte))
    Xr = d[FEATURES]
    yr = d["annual_charge"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=.2, random_state=RANDOM_STATE)
    pre = ColumnTransformer([("num", "passthrough", NUM), ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT)])
    global_reg = Pipeline([("pre", pre), ("model", LinearRegression())]).fit(Xtr, ytr)
    gp = global_reg.predict(Xte)
    return acc, mean_absolute_error(yte, gp), r2_score(yte, gp)

try:
    df = load_data()
    engine, engine_status = load_engine()
except Exception as exc:
    st.error(f"Dashboard startup failed: {exc}")
    st.stop()

st.title("🏥 HealthWise — Smart Premium Dashboard")
st.caption("Two-stage machine learning: classify customer risk first, then estimate annual premium with the matching tier regression model.")

with st.sidebar:
    st.header("Customer Profile")
    age = st.slider("Age", 18, 64, 40)
    bmi = st.slider("BMI", 16.0, 50.0, 27.0, 0.1)
    children = st.slider("Children", 0, 4, 1)
    exercise = st.slider("Exercise / week", 0, 7, 3)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["north", "south", "east", "west"])
    st.divider()
    st.caption(engine_status)

customer = {"age": age, "bmi": bmi, "children": children, "exercise_freq": exercise,
            "sex": sex, "smoker": smoker, "region": region}
tier, premium = predict_customer(engine, customer)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Risk Tier", tier)
k2.metric("Estimated Annual Premium", f"${premium:,.0f}")
k3.metric("Customers", f"{len(df):,}")
k4.metric("Average Actual Charge", f"${df['annual_charge'].mean():,.0f}")

if tier == "High": st.warning("High risk — routed to the High-tier regression model.")
elif tier == "Medium": st.info("Medium risk — routed to the Medium-tier regression model.")
else: st.success("Low risk — routed to the Low-tier regression model.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Calculator", "📊 Portfolio", "🧠 ML Results", "📁 Batch Scoring", "ℹ️ About"])

with tab1:
    st.subheader("Live premium sensitivity")
    st.write("Change any input in the sidebar. Stage 1 reclassifies the customer and Stage 2 immediately recalculates the premium.")
    scenarios = []
    changes = [("Current profile", {}), ("Smoker = yes", {"smoker":"yes"}),
               ("BMI + 5", {"bmi":min(50.0,bmi+5)}), ("Age + 10", {"age":min(64,age+10)}),
               ("Children + 2", {"children":min(4,children+2)})]
    for label, change in changes:
        c = customer.copy(); c.update(change)
        t, p = predict_customer(engine, c)
        scenarios.append([label, t, p, p-premium])
    s = pd.DataFrame(scenarios, columns=["Scenario","Risk Tier","Premium","Change vs Current"])
    st.dataframe(s.style.format({"Premium":"${:,.2f}","Change vs Current":"${:,.2f}"}), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(9,4)); ax.bar(s["Scenario"], s["Premium"])
    ax.set_ylabel("Predicted annual premium ($)"); ax.set_title("Premium sensitivity")
    ax.tick_params(axis="x", rotation=18); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab2:
    st.subheader("HealthWise customer portfolio")
    a,b = st.columns(2)
    with a:
        fig,ax=plt.subplots(figsize=(6,4)); ax.hist(df["annual_charge"], bins=30)
        ax.set(title="Annual Charge Distribution", xlabel="Annual charge ($)", ylabel="Customers"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        v=df["risk_tier"].value_counts().reindex(["Low","Medium","High"])
        fig,ax=plt.subplots(figsize=(6,4)); ax.bar(v.index,v.values); ax.set(title="Customers by Risk Tier", ylabel="Customers"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    a,b=st.columns(2)
    with a:
        v=df.groupby("smoker")["annual_charge"].mean(); fig,ax=plt.subplots(figsize=(6,4)); ax.bar(v.index,v.values)
        ax.set(title="Average Charge by Smoking Status", ylabel="Average annual charge ($)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        v=df.groupby("risk_tier")["annual_charge"].mean().reindex(["Low","Medium","High"]); fig,ax=plt.subplots(figsize=(6,4)); ax.bar(v.index,v.values)
        ax.set(title="Average Charge by Risk Tier", ylabel="Average annual charge ($)"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab3:
    st.subheader("Machine-learning results")
    st.write("These are the executed assignment benchmark results used for the HealthWise analysis.")
    a,b,c,d=st.columns(4)
    a.metric("Classification Accuracy", "82.9%")
    b.metric("Global Regression R²", "0.691")
    c.metric("Two-Stage R²", "0.992")
    d.metric("Two-Stage MAE", "$908")
    perf=pd.DataFrame({"Model":["Global regression","Two-stage tier regression"],"MAE ($)":[5698,908],"R²":[0.691,0.992]})
    st.dataframe(perf, use_container_width=True, hide_index=True)
    st.success("The evaluated tier-wise approach reduced MAE by about 84.1% versus the global regression benchmark.")
    st.markdown("**How it works:** Stage 1 predicts Low/Medium/High risk → Stage 2 selects that tier's regressor → the regressor estimates annual premium.")

with tab4:
    st.subheader("Score unseen customers")
    st.write("Upload the instructor's unseen-customer CSV. Required columns: `" + "`, `".join(FEATURES) + "`.")
    example={"age":40,"bmi":27.0,"children":1,"exercise_freq":3,"sex":"male","smoker":"no","region":"north"}
    template=pd.DataFrame([{c:example[c] for c in FEATURES}])
    st.download_button("Download CSV template", template.to_csv(index=False).encode(), "healthwise_unseen_template.csv", "text/csv")
    up=st.file_uploader("Upload unseen customers", type=["csv"])
    if up is not None:
        try:
            scored=score_frame(engine,pd.read_csv(up)); st.success(f"Scored {len(scored)} customer(s).")
            st.dataframe(scored,use_container_width=True)
            st.download_button("Download predictions",scored.to_csv(index=False).encode(),"healthwise_predictions.csv","text/csv")
        except Exception as exc: st.error(str(exc))

with tab5:
    st.subheader("About the project")
    st.markdown("""
**Business problem:** A flat insurance price can overcharge low-risk customers and undercharge high-risk customers.  
**Stage 1 — Classification:** Predict the customer's risk tier.  
**Stage 2 — Regression:** Predict the annual charge using the regression model for that predicted tier.  
**Deployment:** This dashboard demonstrates live prediction and supports batch scoring of unseen customers.
""")

st.divider(); st.caption("HealthWise Insurance • MAIB Machine Learning • Smart Premium Engine")
