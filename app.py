"""
Mobile Usage Data Analysis System
----------------------------------
An interactive Streamlit dashboard for analyzing smartphone usage patterns,
app engagement, screen time and internet/data consumption.

Run locally:    streamlit run app.py
Deploy online:  Streamlit Community Cloud (see README.md)
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Usage Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# COLUMN NAME CONSTANTS  (must match the CSV header exactly)
# --------------------------------------------------------------------------------
COL_USER_ID = "User ID"
COL_DEVICE = "Device Model"
COL_OS = "Operating System"
COL_APP_USAGE = "App Usage Time (min/day)"
COL_SCREEN_TIME = "Screen On Time (hours/day)"
COL_BATTERY = "Battery Drain (mAh/day)"
COL_APPS_INSTALLED = "Number of Apps Installed"
COL_DATA_USAGE = "Data Usage (MB/day)"
COL_AGE = "Age"
COL_GENDER = "Gender"
COL_BEHAVIOR = "User Behavior Class"

NUMERIC_COLS = [
    COL_APP_USAGE, COL_SCREEN_TIME, COL_BATTERY,
    COL_APPS_INSTALLED, COL_DATA_USAGE, COL_AGE, COL_BEHAVIOR,
]

BEHAVIOR_LABELS = {
    1: "1 · Light User",
    2: "2 · Below Average",
    3: "3 · Average User",
    4: "4 · Above Average",
    5: "5 · Heavy / Extreme User",
}

# --------------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING  (Module 1: Data Collection & Preprocessing)
# --------------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    required = [
        COL_USER_ID, COL_DEVICE, COL_OS, COL_APP_USAGE, COL_SCREEN_TIME,
        COL_BATTERY, COL_APPS_INSTALLED, COL_DATA_USAGE, COL_AGE,
        COL_GENDER, COL_BEHAVIOR,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=required)
    for col in [COL_APP_USAGE, COL_BATTERY, COL_APPS_INSTALLED,
                COL_DATA_USAGE, COL_AGE, COL_BEHAVIOR]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[COL_SCREEN_TIME] = pd.to_numeric(df[COL_SCREEN_TIME], errors="coerce")
    df = df.dropna(subset=required)

    df[COL_DEVICE] = df[COL_DEVICE].astype(str).str.strip()
    df[COL_OS] = df[COL_OS].astype(str).str.strip()
    df[COL_GENDER] = df[COL_GENDER].astype(str).str.strip()

    # Derived / engineered features used across the dashboard
    df["Behavior Label"] = df[COL_BEHAVIOR].map(BEHAVIOR_LABELS)
    df["Age Group"] = pd.cut(
        df[COL_AGE],
        bins=[0, 25, 35, 45, 55, 200],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    df["Screen Time (min/day)"] = df[COL_SCREEN_TIME] * 60
    df["App Usage Share (%)"] = (
        df[COL_APP_USAGE] / df["Screen Time (min/day)"] * 100
    ).clip(upper=100).round(1)
    df["Data per App (MB/day)"] = (
        df[COL_DATA_USAGE] / df[COL_APPS_INSTALLED]
    ).round(2)
    df["Projected Weekly Screen Time (hrs)"] = (df[COL_SCREEN_TIME] * 7).round(1)
    df["Projected Monthly Data Usage (GB)"] = (
        df[COL_DATA_USAGE] * 30 / 1024
    ).round(2)
    # Simple 0-100 composite engagement index (min-max normalised average
    # of screen time, app usage time and data usage)
    for raw, norm in [
        (COL_SCREEN_TIME, "_n_screen"),
        (COL_APP_USAGE, "_n_app"),
        (COL_DATA_USAGE, "_n_data"),
    ]:
        rng = df[raw].max() - df[raw].min()
        df[norm] = (df[raw] - df[raw].min()) / rng if rng else 0.0
    df["Digital Engagement Index"] = (
        (df["_n_screen"] + df["_n_app"] + df["_n_data"]) / 3 * 100
    ).round(1)
    df = df.drop(columns=["_n_screen", "_n_app", "_n_data"])

    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    feature_cols_num = [
        COL_APP_USAGE, COL_SCREEN_TIME, COL_BATTERY,
        COL_APPS_INSTALLED, COL_DATA_USAGE, COL_AGE,
    ]
    feature_cols_cat = [COL_DEVICE, COL_OS, COL_GENDER]

    X = df[feature_cols_num + feature_cols_cat]
    y = df[COL_BEHAVIOR]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ],
        remainder="passthrough",
    )

    model = Pipeline(steps=[
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    # Feature importance
    ohe_names = list(
        model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(feature_cols_cat)
    )
    all_names = ohe_names + feature_cols_num
    importances = model.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"Feature": all_names, "Importance": importances}) \
        .sort_values("Importance", ascending=False)

    return model, acc, fi, feature_cols_num, feature_cols_cat


def kpi_row(df: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Users", f"{len(df):,}")
    c2.metric("🖥️ Avg Screen Time", f"{df[COL_SCREEN_TIME].mean():.1f} hrs/day")
    c3.metric("📲 Avg App Usage", f"{df[COL_APP_USAGE].mean():.0f} min/day")
    c4.metric("🌐 Avg Data Usage", f"{df[COL_DATA_USAGE].mean():.0f} MB/day")
    c5.metric("📦 Avg Apps Installed", f"{df[COL_APPS_INSTALLED].mean():.0f}")


def df_download_button(df: pd.DataFrame, label: str, file_name: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=file_name, mime="text/csv")


# --------------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE + FILTERS
# --------------------------------------------------------------------------------
st.sidebar.title("📱 Mobile Usage Analytics")
st.sidebar.caption("Mobile Usage Data Analysis System")

default_path = "datamobile_usage.csv"
uploaded = st.sidebar.file_uploader("Upload usage dataset (CSV)", type=["csv"])

try:
    if uploaded is not None:
        raw_df = load_data(uploaded)
    else:
        raw_df = load_data(default_path)
except FileNotFoundError:
    st.sidebar.error(
        "Default dataset not found. Please upload a CSV file with mobile "
        "usage data to continue."
    )
    st.stop()
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()

st.sidebar.success(f"Loaded {len(raw_df):,} records")

st.sidebar.header("🔎 Filters")

devices = st.sidebar.multiselect(
    "Device Model", sorted(raw_df[COL_DEVICE].unique()),
    default=sorted(raw_df[COL_DEVICE].unique()),
)
os_choice = st.sidebar.multiselect(
    "Operating System", sorted(raw_df[COL_OS].unique()),
    default=sorted(raw_df[COL_OS].unique()),
)
gender_choice = st.sidebar.multiselect(
    "Gender", sorted(raw_df[COL_GENDER].unique()),
    default=sorted(raw_df[COL_GENDER].unique()),
)
behavior_choice = st.sidebar.multiselect(
    "Behavior Class",
    options=sorted(raw_df[COL_BEHAVIOR].unique()),
    default=sorted(raw_df[COL_BEHAVIOR].unique()),
    format_func=lambda x: BEHAVIOR_LABELS.get(x, str(x)),
)
age_min, age_max = int(raw_df[COL_AGE].min()), int(raw_df[COL_AGE].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

screen_min, screen_max = float(raw_df[COL_SCREEN_TIME].min()), float(raw_df[COL_SCREEN_TIME].max())
screen_range = st.sidebar.slider(
    "Screen On Time (hrs/day)", screen_min, screen_max, (screen_min, screen_max)
)

if st.sidebar.button("♻️ Reset Filters"):
    st.rerun()

st.sidebar.caption(
    "💡 Tip: use the ⋮ menu (top-right) → Settings → Theme to switch "
    "between Light / Dark mode."
)

df = raw_df[
    raw_df[COL_DEVICE].isin(devices)
    & raw_df[COL_OS].isin(os_choice)
    & raw_df[COL_GENDER].isin(gender_choice)
    & raw_df[COL_BEHAVIOR].isin(behavior_choice)
    & raw_df[COL_AGE].between(age_range[0], age_range[1])
    & raw_df[COL_SCREEN_TIME].between(screen_range[0], screen_range[1])
].copy()

if df.empty:
    st.warning("No records match the selected filters. Please widen your filters.")
    st.stop()

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.title("📱 Mobile Usage Data Analysis System")
st.caption(
    "Smartphone usage patterns · App engagement · Screen time · "
    "Internet & data consumption · User behavior analytics"
)
kpi_row(df)
st.divider()

# --------------------------------------------------------------------------------
# TABS  (mirrors the suggested project modules)
# --------------------------------------------------------------------------------
tab_overview, tab_screen, tab_app, tab_internet, tab_behavior, tab_predict, tab_export = st.tabs(
    [
        "🏠 Overview",
        "🖥️ Screen Time",
        "📲 App Usage",
        "🌐 Internet & Data",
        "🧠 Behavior Insights",
        "🔮 Usage Prediction",
        "📥 Reports",
    ]
)

# ---------------------------------------------------------------- Overview ----
with tab_overview:
    st.subheader("Data Management")
    st.caption(
        "This dataset contains one summarized daily-usage record per user "
        "(no per-app or per-hour logs), so the dashboard focuses on "
        "device-level, demographic and behavioral analytics rather than "
        "minute-by-minute timelines."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(df, names=COL_DEVICE, title="Users by Device Model", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(df, names=COL_OS, title="Users by Operating System", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.pie(df, names=COL_GENDER, title="Users by Gender", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        df, x=COL_AGE, nbins=20, title="Age Distribution", color=COL_GENDER,
        barmode="overlay", opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Dataset")
    st.dataframe(df.drop(columns=["Behavior Label"]), use_container_width=True, height=320)

    with st.expander("📈 Summary Statistics"):
        st.dataframe(df[NUMERIC_COLS].describe().round(2), use_container_width=True)

# ------------------------------------------------------------- Screen Time ----
with tab_screen:
    st.subheader("Screen Time Analytics")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x=COL_SCREEN_TIME, nbins=25,
            title="Daily Screen Time Distribution (hrs/day)",
            color_discrete_sequence=["#6C5CE7"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        avg_by_device = df.groupby(COL_DEVICE, as_index=False)[COL_SCREEN_TIME].mean()
        fig = px.bar(
            avg_by_device.sort_values(COL_SCREEN_TIME), x=COL_DEVICE, y=COL_SCREEN_TIME,
            title="Average Screen Time by Device", color=COL_DEVICE,
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            df, x="Behavior Label", y=COL_SCREEN_TIME, color="Behavior Label",
            title="Screen Time by Behavior Class",
            category_orders={"Behavior Label": [BEHAVIOR_LABELS[i] for i in range(1, 6)]},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(
            df, x=COL_AGE, y=COL_SCREEN_TIME, color=COL_GENDER,
            title="Screen Time vs Age", trendline="ols",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df, x=COL_SCREEN_TIME, y=COL_BATTERY, color=COL_OS,
        title="Screen Time vs Battery Drain", trendline="ols", opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### ⏱️ Projected Weekly / Monthly Usage (extrapolated from daily averages)")
    p1, p2, p3 = st.columns(3)
    p1.metric("Avg Daily Screen Time", f"{df[COL_SCREEN_TIME].mean():.1f} hrs")
    p2.metric("Projected Weekly", f"{df['Projected Weekly Screen Time (hrs)'].mean():.1f} hrs")
    p3.metric("Projected Monthly", f"{df[COL_SCREEN_TIME].mean() * 30:.0f} hrs")

    st.markdown("##### 🏆 Top 10 Users by Screen Time")
    top10 = df.nlargest(10, COL_SCREEN_TIME)[
        [COL_USER_ID, COL_DEVICE, COL_OS, COL_SCREEN_TIME, COL_APP_USAGE, COL_BEHAVIOR]
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------- App Use ----
with tab_app:
    st.subheader("App Usage Analysis")
    st.caption(
        "The dataset reports total daily app-usage minutes and number of "
        "apps installed per user (not a per-app breakdown). The metrics "
        "below analyze engagement at the device / behavior-class level."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x=COL_APPS_INSTALLED, nbins=25,
            title="Number of Apps Installed — Distribution",
            color_discrete_sequence=["#00B894"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            df, x=COL_APP_USAGE, nbins=25,
            title="Daily App Usage Time Distribution (min/day)",
            color_discrete_sequence=["#0984E3"],
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        avg_by_device = df.groupby(COL_DEVICE, as_index=False)[COL_APP_USAGE].mean()
        fig = px.bar(
            avg_by_device.sort_values(COL_APP_USAGE), x=COL_DEVICE, y=COL_APP_USAGE,
            title="Average App Usage Time by Device", color=COL_DEVICE,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(
            df, x=COL_APPS_INSTALLED, y=COL_APP_USAGE, color="Behavior Label",
            title="Apps Installed vs Daily App Usage Time", opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df, x="Behavior Label", y="App Usage Share (%)", color="Behavior Label",
        title="App-Usage Share of Total Screen Time, by Behavior Class (%)",
        category_orders={"Behavior Label": [BEHAVIOR_LABELS[i] for i in range(1, 6)]},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 🏆 Top 10 Most App-Engaged Users")
    top10 = df.nlargest(10, COL_APP_USAGE)[
        [COL_USER_ID, COL_DEVICE, COL_APP_USAGE, COL_APPS_INSTALLED, "App Usage Share (%)"]
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ------------------------------------------------------------- Internet/Data ----
with tab_internet:
    st.subheader("Internet & Data Consumption Tracking")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df, x=COL_DATA_USAGE, nbins=25,
            title="Daily Data Usage Distribution (MB/day)",
            color_discrete_sequence=["#E17055"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        avg_by_os = df.groupby(COL_OS, as_index=False)[COL_DATA_USAGE].mean()
        fig = px.bar(
            avg_by_os, x=COL_OS, y=COL_DATA_USAGE, color=COL_OS,
            title="Average Data Usage by Operating System",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        avg_by_device = df.groupby(COL_DEVICE, as_index=False)[COL_DATA_USAGE].mean()
        fig = px.bar(
            avg_by_device.sort_values(COL_DATA_USAGE), x=COL_DEVICE, y=COL_DATA_USAGE,
            title="Average Data Usage by Device", color=COL_DEVICE,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(
            df, x=COL_APPS_INSTALLED, y=COL_DATA_USAGE, color=COL_OS,
            title="Apps Installed vs Data Usage", trendline="ols", opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df.groupby(COL_DEVICE, as_index=False)["Data per App (MB/day)"].mean()
        .sort_values("Data per App (MB/day)"),
        x=COL_DEVICE, y="Data per App (MB/day)", color=COL_DEVICE,
        title="Average Data Consumption per Installed App (MB/day)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📡 Projected Monthly Internet Consumption")
    p1, p2 = st.columns(2)
    p1.metric("Avg Daily Data Usage", f"{df[COL_DATA_USAGE].mean():.0f} MB")
    p2.metric("Projected Monthly", f"{df['Projected Monthly Data Usage (GB)'].mean():.2f} GB")

    st.markdown("##### 🏆 Top 10 Highest Data-Consuming Users")
    top10 = df.nlargest(10, COL_DATA_USAGE)[
        [COL_USER_ID, COL_DEVICE, COL_OS, COL_DATA_USAGE, COL_APPS_INSTALLED, "Data per App (MB/day)"]
    ]
    st.dataframe(top10, use_container_width=True, hide_index=True)

# ------------------------------------------------------------ Behavior Insights ----
with tab_behavior:
    st.subheader("User Behavior Insights")

    c1, c2 = st.columns(2)
    with c1:
        counts = df["Behavior Label"].value_counts().reindex(
            [BEHAVIOR_LABELS[i] for i in range(1, 6)]
        ).fillna(0)
        fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            title="User Distribution by Behavior Class",
            labels={"x": "Behavior Class", "y": "Number of Users"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr = df[NUMERIC_COLS].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Heatmap — Usage Metrics",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        age_grp = df.groupby("Age Group", observed=True, as_index=False)[COL_BEHAVIOR].mean()
        fig = px.bar(
            age_grp, x="Age Group", y=COL_BEHAVIOR,
            title="Average Behavior Class by Age Group", color="Age Group",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        gender_grp = df.groupby(COL_GENDER, as_index=False)[
            [COL_SCREEN_TIME, COL_APP_USAGE, COL_DATA_USAGE]
        ].mean()
        gender_long = gender_grp.melt(id_vars=COL_GENDER, var_name="Metric", value_name="Value")
        fig = px.bar(
            gender_long, x="Metric", y="Value", color=COL_GENDER, barmode="group",
            title="Average Usage Metrics by Gender",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 🌱 Digital Engagement Index")
    st.caption(
        "A 0-100 composite score combining normalized screen time, app "
        "usage time and data usage — higher means heavier digital usage. "
        "This is a relative indicator within the current dataset, not a "
        "medical or clinical measure."
    )
    fig = px.histogram(
        df, x="Digital Engagement Index", nbins=25, color="Behavior Label",
        title="Digital Engagement Index Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_eng = df["Digital Engagement Index"].mean()
    if avg_eng >= 70:
        tier_msg = "🔴 The filtered group shows **high** overall digital engagement."
    elif avg_eng >= 40:
        tier_msg = "🟠 The filtered group shows **moderate** digital engagement."
    else:
        tier_msg = "🟢 The filtered group shows **low** digital engagement."
    st.info(f"Average Engagement Index: **{avg_eng:.1f}/100** — {tier_msg}")

    st.markdown("##### 📋 Average Metrics by Behavior Class")
    summary = df.groupby("Behavior Label", as_index=False)[
        [COL_SCREEN_TIME, COL_APP_USAGE, COL_DATA_USAGE, COL_APPS_INSTALLED, COL_BATTERY]
    ].mean().round(1)
    st.dataframe(summary, use_container_width=True, hide_index=True)

# -------------------------------------------------------------- Prediction ----
with tab_predict:
    st.subheader("🔮 AI-Based Usage Class Prediction")
    st.caption(
        "A Random Forest model trained on the current dataset predicts a "
        "user's Behavior Class (1=Light … 5=Extreme) from their usage "
        "profile. Adjust the sliders below to see a live prediction."
    )

    with st.spinner("Training model on the current dataset..."):
        model, acc, fi, num_feats, cat_feats = train_model(raw_df)

    st.success(f"Model trained — hold-out test accuracy: **{acc * 100:.1f}%**")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(
            fi.head(10), x="Importance", y="Feature", orientation="h",
            title="Top Feature Importances",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Try it yourself")
        in_device = st.selectbox("Device Model", sorted(raw_df[COL_DEVICE].unique()))
        in_os = st.selectbox("Operating System", sorted(raw_df[COL_OS].unique()))
        in_gender = st.selectbox("Gender", sorted(raw_df[COL_GENDER].unique()))
        in_age = st.slider("Age", int(raw_df[COL_AGE].min()), int(raw_df[COL_AGE].max()), 28)
        in_screen = st.slider("Screen On Time (hrs/day)", 0.0, 14.0, 5.0, 0.1)
        in_app = st.slider("App Usage Time (min/day)", 0, 700, 200)
        in_apps = st.slider("Number of Apps Installed", 0, 120, 45)
        in_battery = st.slider("Battery Drain (mAh/day)", 0, 4000, 1500)
        in_data = st.slider("Data Usage (MB/day)", 0, 3000, 900)

        if st.button("🔮 Predict Usage Class", type="primary"):
            row = pd.DataFrame([{
                COL_APP_USAGE: in_app,
                COL_SCREEN_TIME: in_screen,
                COL_BATTERY: in_battery,
                COL_APPS_INSTALLED: in_apps,
                COL_DATA_USAGE: in_data,
                COL_AGE: in_age,
                COL_DEVICE: in_device,
                COL_OS: in_os,
                COL_GENDER: in_gender,
            }])
            pred = int(model.predict(row)[0])
            st.metric("Predicted Behavior Class", BEHAVIOR_LABELS.get(pred, str(pred)))

# ------------------------------------------------------------------- Reports ----
with tab_export:
    st.subheader("📥 Exportable Analytics Reports")

    df_download_button(df, "⬇️ Download Filtered Data (CSV)", "filtered_mobile_usage.csv")

    summary_lines = [
        "MOBILE USAGE DATA ANALYSIS — SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Records analyzed: {len(df)}",
        "",
        "--- Screen Time ---",
        f"Average daily screen time: {df[COL_SCREEN_TIME].mean():.2f} hrs",
        f"Max daily screen time: {df[COL_SCREEN_TIME].max():.2f} hrs",
        "",
        "--- App Usage ---",
        f"Average app usage time: {df[COL_APP_USAGE].mean():.1f} min/day",
        f"Average apps installed: {df[COL_APPS_INSTALLED].mean():.1f}",
        "",
        "--- Internet & Data Usage ---",
        f"Average data usage: {df[COL_DATA_USAGE].mean():.1f} MB/day",
        f"Projected avg monthly data usage: {df['Projected Monthly Data Usage (GB)'].mean():.2f} GB",
        "",
        "--- Behavior ---",
        f"Average Digital Engagement Index: {df['Digital Engagement Index'].mean():.1f}/100",
        "Behavior class distribution:",
    ]
    for i in range(1, 6):
        n = int((df[COL_BEHAVIOR] == i).sum())
        summary_lines.append(f"  {BEHAVIOR_LABELS[i]}: {n} users")

    report_text = "\n".join(summary_lines)
    st.text_area("Report Preview", report_text, height=320)
    st.download_button(
        "⬇️ Download Summary Report (TXT)",
        report_text.encode("utf-8"),
        file_name="mobile_usage_summary_report.txt",
        mime="text/plain",
    )

st.divider()
st.caption(
    "Mobile Usage Data Analysis System · Built with Streamlit, Pandas & "
    "Plotly · Internship Analytics Project"
)
