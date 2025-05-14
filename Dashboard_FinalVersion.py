# =========================== Import Packages ===========================
import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy import stats

# =========================== Load and Prepare Data ===========================
df_numeric = pd.read_csv('XAI_data.csv')
non_binary = df_numeric.loc[:, df_numeric.apply(lambda x: x.nunique()) >= 3]
df_outlier_removal = df_numeric[list(non_binary.columns)]
df_outlier_removal = df_outlier_removal[(np.abs(stats.zscore(df_outlier_removal)) < 3).all(axis=1)]
df_binary = df_numeric.drop(columns=list(df_outlier_removal.columns))
df_outlier = df_outlier_removal.merge(df_binary, left_index=True, right_index=True, how='inner').reset_index(drop=True)

X_outlier = df_outlier.drop(columns='Price')
y_outlier = df_outlier['Price']

train_X, test_X, train_y, test_y = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=3)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

train_X = pd.DataFrame(train_X, columns=X_outlier.columns)
test_X = pd.DataFrame(test_X, columns=X_outlier.columns)

# Train XGB model
xgb_model = XGBRegressor(n_estimators=80, max_depth=5, eta=0.1)
xgb_model.fit(train_X, train_y)

# Load explainers
@st.cache_resource
def get_explainers():
    xgb_shap = shap.Explainer(xgb_model, test_X)
    xgb_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        mode='regression',
        verbose=False
    )
    return xgb_shap, xgb_lime

xgb_shap, xgb_lime = get_explainers()

# =========================== Streamlit Layout ===========================
st.set_page_config(page_title="Housing Price Explanation", layout="wide")
st.title("🏠 Housing Price Explanation Dashboard")

sample_idx = st.sidebar.slider("Select a sample index", 0, len(test_X)-1, 0)
X_sample = test_X.iloc[[sample_idx]]

# =========================== Feature Overview ===========================
with st.expander("📌 Selected Index - Feature Overview (Original Values)", expanded=True):
    X_sample_original = X_outlier.iloc[[sample_idx]].T
    X_sample_original.columns = ['Original Value']
    X_sample_original.index.name = 'Feature'
    
    st.dataframe(
        X_sample_original.style.format(precision=2),
        use_container_width=True,
        height=200
    )


# =========================== SHAP & LIME Explanation ===========================

# add tooltip for SHAP and LIME
st.markdown("""
<style>
.tooltip {
  position: relative;
  display: inline-block;
  cursor: help;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 280px;
  background-color: #f9f9f9;
  color: #333;
  text-align: left;
  border-radius: 6px;
  padding: 8px;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 0%;
  margin-left: -20px;
  opacity: 0;
  transition: opacity 0.3s;
  border: 1px solid #ccc;
  box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
  font-size: 18px;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)

st.header("🧠 Explanation Comparison for XGBRegressor")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style='padding: 10px; border: 2px solid #4A90E2; border-radius: 10px; background-color: #F0F8FF'>
        <h4 style='color: #1f77b4;'>🔷 SHAP Explanation 
        <span class="tooltip">ℹ️
            <span class="tooltiptext">
            SHAP explains how each feature contributes to pushing the model prediction higher or lower.
            </span>
        </span>
        </h4>
        """,
        unsafe_allow_html=True
    )

    shap_values = xgb_shap(X_sample)

    st.markdown("""
    <h5>📈 Waterfall Plot 
    <span class="tooltip">ℹ️
        <span class="tooltiptext">
        This waterfall plot shows how much each feature contributes to the final prediction for this specific sample.
        </span>
    </span>
    </h5>
    """, unsafe_allow_html=True)
    fig1, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    st.markdown("""
    <h5>📊 SHAP Summary Plot (entire test set) 
    <span class="tooltip">ℹ️
        <span class="tooltiptext">
        This summary plot shows how all features contribute across the test set, sorted by importance. The color indicates feature value.
        </span>
    </span>
    </h5>
    """, unsafe_allow_html=True)
    shap_values_all = xgb_shap(test_X)
    fig2, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values_all, test_X, show=False)
    st.pyplot(fig2)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div style='padding: 10px; border: 2px solid #FFA500; border-radius: 10px; background-color: #FFF8DC'>
        <h4 style='color: #ff7f0e;'>🟠 LIME Explanation 
        <span class="tooltip">ℹ️
            <span class="tooltiptext">
            LIME explains the prediction by approximating the model locally with an interpretable linear model.
            </span>
        </span>
        </h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <h5>📋 LIME Feature Impact 
    <span class="tooltip">ℹ️
        <span class="tooltiptext">
        The LIME explanation shows how local feature conditions influence the prediction of this specific sample.
        </span>
    </span>
    </h5>
    """, unsafe_allow_html=True)

    lime_exp = xgb_lime.explain_instance(
        data_row=np.array(X_sample.iloc[0]),
        predict_fn=xgb_model.predict
    )
    html = lime_exp.as_html()
    components.html(html, height=800, scrolling=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================== Feature Comparison & Overlap ===========================
st.subheader("📊 Top 5 Feature Comparison")

# SHAP top features
shap_top = pd.DataFrame({
    'feature': test_X.columns,
    'shap_value': np.abs(shap_values.values[0])
}).sort_values(by='shap_value', ascending=False)['feature'].head(5).tolist()

# LIME top features (with intervals like "-0.68 < Volume in m3 ≤ -0.04")
import re
lime_top_raw = [x[0] for x in lime_exp.as_list()[:5]]

# function to extract feature name from LIME output
def extract_feature_name(feature_str):
    # match "Volume in m3" from things like "-0.68 < Volume in m3 ≤ -0.04"
    match = re.findall(r"[a-zA-Z_][a-zA-Z0-9_ ]*", feature_str)
    return match[-1].strip() if match else feature_str.strip()

lime_top_clean = [extract_feature_name(f) for f in lime_top_raw]

# comparison of top features
comparison_df = pd.DataFrame({
    "Rank": list(range(1, 6)),
    "SHAP Top Features": shap_top,
    "LIME Top Features": lime_top_raw
   # "LIME Parsed Feature Name": lime_top_clean
})
st.table(comparison_df)

# calculate overlap
overlap = set(shap_top) & set(lime_top_clean)
st.subheader("🔁 Overlap Analysis")
if overlap:
    st.write(f"**Overlap features:** `{', '.join(overlap)}`")
    st.success(f"✅ {len(overlap)} out of 5 features overlap")
else:
    st.warning("❌ No overlap found between SHAP and LIME top features.")


# =========================== User Feedback Section ===========================
st.sidebar.markdown("## 💬 User Feedback")

understandable = st.sidebar.radio(
    "1️⃣ Which explanation do you find easier to understand?",
    ["SHAP", "LIME", "Both equally", "Neither"]
)

trusted_explainer = st.sidebar.radio(
    "2️⃣ Which explanation do you trust more?",
    ["SHAP", "LIME", "Both equally", "Neither"]
)

feedback = st.sidebar.text_area("🗣️ Any additional comments or suggestions?")

if st.sidebar.button("📩 Submit Feedback"):
    st.sidebar.success("✅ Thank you for your feedback!")
    st.write("### 🙌 Feedback Summary")
    st.write(f"- Easier to understand: **{understandable}**")
    st.write(f"- More trusted explanation: **{trusted_explainer}**")
    st.write(f"- Additional Comments:\n{feedback if feedback else 'None'}")


# =========================== Run the dashboard ===========================
# streamlit run Dashboard_FinalVersion.py