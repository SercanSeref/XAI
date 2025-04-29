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
from xgboost import XGBRegressor
from scipy import stats

# ========== Data Preprocessing Functions ==========
def splitting_data(X, y):
    column_names = list(X.columns)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=3)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.25, random_state=3)
    return train_X, valid_X, test_X, train_y, test_y, valid_y, column_names

def scale_data(scalar, train_X, valid_X, test_X, column_names):
    scaler = scalar.fit(train_X)
    train_X = scaler.transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)
    train_X = pd.DataFrame(train_X, columns=column_names)
    valid_X = pd.DataFrame(valid_X, columns=column_names)
    test_X = pd.DataFrame(test_X, columns=column_names)
    return train_X, valid_X, test_X

# ========== Load and Prepare Data ==========
df_numeric = pd.read_csv('XAI_data.csv')
non_binary = df_numeric.loc[:, df_numeric.apply(lambda x: x.nunique()) >= 3]
df_outlier_removal = df_numeric[list(non_binary.columns)]
df_outlier_removal = df_outlier_removal[(np.abs(stats.zscore(df_outlier_removal)) < 3).all(axis=1)]
df_binary = df_numeric.drop(columns=list(df_outlier_removal.columns))
df_outlier = df_outlier_removal.merge(df_binary, left_index=True, right_index=True, how='inner').reset_index(drop=True)
X_outlier = df_outlier.drop(columns='Price')
y_outlier = df_outlier['Price']

train_X, valid_X, test_X, train_y, test_y, valid_y, column_names = splitting_data(X_outlier, y_outlier)
train_X, valid_X, test_X = scale_data(StandardScaler(), train_X, valid_X, test_X, column_names)

model = XGBRegressor(n_estimators=80, max_depth=5, eta=0.1, silent=True)
model.fit(train_X, train_y)

# ========== Streamlit Layout Setup ==========
st.set_page_config(page_title="XAI Dashboard", layout="wide")
st.title("🔍 SHAP vs LIME Explanation Dashboard")

page = st.sidebar.radio("Navigation", ["SHAP", "LIME", "Comparison"])
sample_idx = st.sidebar.slider("Select a sample index", 0, len(test_X)-1, 0)
X_sample = test_X.iloc[[sample_idx]]

@st.cache_resource
def get_shap_explainer():
    return shap.Explainer(model, test_X)

# ========== SHAP Page ==========
if page == "SHAP":
    st.header("SHAP Explanation for Selected Sample")
    explainer = get_shap_explainer()
    shap_values = explainer(X_sample)

    st.subheader("Waterfall Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.subheader("Force Plot")
    force_html = shap.plots.force(shap_values[0], matplotlib=False)
    st.components.v1.html(shap.getjs() + force_html.html(), height=300)

    st.subheader("Summary Plot (based on whole test set)")
    shap_values_all = explainer(test_X)
    fig_summary, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values_all, test_X, show=False)
    st.pyplot(fig_summary)

# ========== LIME Page ==========
elif page == "LIME":
    st.header("LIME Explanation for Selected Sample")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        mode='regression',
        verbose=False
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=np.array(X_sample.iloc[0]),
        predict_fn=model.predict
    )
    html = lime_exp.as_html()
    components.html(html, height=800, scrolling=True)

# ========== Comparison Page ==========
elif page == "Comparison":
    st.header("🔍 SHAP vs LIME Comparison")
    explainer = get_shap_explainer()
    shap_values_all = explainer(test_X)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        mode='regression',
        verbose=False
    )
    lime_exp = lime_explainer.explain_instance(
        data_row=np.array(X_sample.iloc[0]),
        predict_fn=model.predict
    )

    # Top 5 features by SHAP absolute values
    shap_values = shap_values_all[sample_idx]
    shap_feature_imp = pd.DataFrame({
        'feature': test_X.columns,
        'shap_value': np.abs(shap_values.values[0])
    }).sort_values(by='shap_value', ascending=False)
    shap_top = shap_feature_imp['feature'].head(5).tolist()

    # Top 5 features by LIME
    lime_top = [x[0] for x in lime_exp.as_list()[:5]]

    # Price prediction comparison
    shap_pred = model.predict(X_sample)[0]
    lime_pred = np.array(lime_exp.predict_proba())[0] if hasattr(lime_exp, 'predict_proba') else shap_pred

    st.subheader("💡 Price Prediction")
    st.write(f"**SHAP predicted price:** {shap_pred:.2f}")
    st.write(f"**LIME predicted price:** {lime_pred:.2f}")

    # Display top features comparison
    st.subheader("📊 Top 5 Feature Comparison")
    comparison_df = pd.DataFrame({
        "Rank": list(range(1, 6)),
        "SHAP Top Features": shap_top,
        "LIME Top Features": lime_top
    })
    st.table(comparison_df)

    # Overlap analysis
    overlap = set(shap_top) & set(lime_top)
    st.subheader("🔁 Overlap Analysis")
    st.write(f"**Overlap features:** `{', '.join(overlap)}`")
    st.success(f"✅ {len(overlap)} out of 5 features overlap")

# ========== Feedback Section ==========
st.sidebar.markdown("## 💬 User Feedback")
st.sidebar.radio("Which explanation is easier to understand?", ["SHAP", "LIME", "Both equally", "Neither"])
st.sidebar.radio("Which explanation do you trust more?", ["SHAP", "LIME", "Both equally", "Neither"])
feedback = st.sidebar.text_area("Any comments or suggestions?")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("✅ Thank you for your feedback!")

# ================= Run the dashboard ===================
# streamlit run xai_dashboard.py