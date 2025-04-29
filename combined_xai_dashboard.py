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
import torch
import torch.nn as nn
import os
from scipy import stats
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel
import seaborn as sns


# =========================== Neural Network Model Definition ===========================
size_hidden1 = 256
size_hidden2 = 128
size_hidden3 = 64
size_hidden4 = 1
input_dim = 30  # Your feature dimension

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, size_hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.lin4 = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, input):
        x = self.dropout1(self.relu1(self.lin1(input)))
        x = self.dropout2(self.relu2(self.lin2(x)))
        x = self.dropout3(self.relu3(self.lin3(x)))
        return self.lin4(x)

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
xgb_model = XGBRegressor(n_estimators=80, max_depth=5, eta=0.1, silent=True)
xgb_model.fit(train_X, train_y)

# Load pretrained NN model
nn_model = Model()
SAVED_MODEL_PATH = 'california_model.pt'
if os.path.exists(SAVED_MODEL_PATH):
    nn_model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=torch.device('cpu')))
    nn_model.eval()

# Prediction function for PyTorch model
def predict_with_nn(input_array):
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    with torch.no_grad():
        output = nn_model(input_tensor).numpy().flatten()
    return output

# =========================== Streamlit Layout Setup ===========================
st.set_page_config(page_title="XAI Dashboard", layout="wide")
st.title("🔍 Model Explanation Dashboard")

model_choice = st.sidebar.selectbox("Select Model", ["XGBRegressor", "Neural Network"])
sample_idx = st.sidebar.slider("Select a sample index", 0, len(test_X)-1, 0)
X_sample = test_X.iloc[[sample_idx]]

# Load explainers
@st.cache_resource
def get_explainers():
    xgb_shap = shap.Explainer(xgb_model, test_X)
    nn_shap = shap.Explainer(lambda x: predict_with_nn(x), np.array(test_X))

    xgb_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        mode='regression',
        verbose=False
    )
    nn_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(train_X),
        feature_names=train_X.columns,
        mode='regression',
        verbose=False
    )
    return xgb_shap, nn_shap, xgb_lime, nn_lime

xgb_shap, nn_shap, xgb_lime, nn_lime = get_explainers()

# Set model and prediction function
if model_choice == "XGBRegressor":
    model = xgb_model
    shap_explainer = xgb_shap
    lime_explainer = xgb_lime
    predict_fn = model.predict
else:
    model = nn_model
    shap_explainer = nn_shap
    lime_explainer = nn_lime
    predict_fn = predict_with_nn

# =========================== Main Dashboard ===========================
st.header(f"Explanation Comparison for {model_choice}")

# 2列展示 SHAP & LIME
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔹 SHAP Explanation")
    shap_values = shap_explainer(X_sample)

    fig1, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig1)

    st.subheader("🔹 SHAP Summary (whole test set)")
    shap_values_all = shap_explainer(test_X)
    fig2, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values_all, test_X, show=False)
    st.pyplot(fig2)

with col2:
    st.subheader("🔸 LIME Explanation")
    lime_exp = lime_explainer.explain_instance(
        data_row=np.array(X_sample.iloc[0]),
        predict_fn=predict_fn
    )
    html = lime_exp.as_html()
    components.html(html, height=800, scrolling=True)

# =========================== Gradient-based Explanation (NN only) ===========================
if model_choice == "Neural Network":
    st.subheader("Gradient-based Explanation Methods (NN Only)")
    grad_col1, grad_col2, grad_col3 = st.columns(3)

    input_tensor = torch.tensor(np.array(X_sample), dtype=torch.float32, requires_grad=True)

    # Integrated Gradients
    with grad_col1:
        st.markdown("**Integrated Gradients**")
        ig = IntegratedGradients(nn_model)
        attr_ig, delta = ig.attribute(input_tensor, target=None, return_convergence_delta=True)
        attr_ig = attr_ig.detach().numpy().flatten()

        fig_ig, ax = plt.subplots(figsize=(6, 4))
        sorted_idx = np.argsort(np.abs(attr_ig))[::-1][:10]
        ax.barh(np.array(test_X.columns)[sorted_idx], attr_ig[sorted_idx])
        ax.set_xlabel('Attribution')
        ax.invert_yaxis()
        st.pyplot(fig_ig)

     #SmoothGrad via NoiseTunnel + IntegratedGradients
    with grad_col2:
        st.markdown("**SmoothGrad (via NoiseTunnel + IG)**")
        ig = IntegratedGradients(nn_model)
        nt = NoiseTunnel(ig)

        attr_sg = nt.attribute(
            input_tensor,
            nt_type='smoothgrad',
            nt_samples=50, 
            target=None
        )
        attr_sg = attr_sg.detach().numpy().flatten()

        fig_sg, ax = plt.subplots(figsize=(6, 4))
        sorted_idx = np.argsort(np.abs(attr_sg))[::-1][:10]
        ax.barh(np.array(test_X.columns)[sorted_idx], attr_sg[sorted_idx])
        ax.set_xlabel('Attribution')
        ax.set_title("Top 10 Features (SmoothGrad)")
        ax.invert_yaxis()
        st.pyplot(fig_sg)

    # GradientSHAP
    with grad_col3:
        st.markdown("**GradientSHAP**")
        baseline_dist = torch.cat([input_tensor * 0, input_tensor * 1])
        gs = GradientShap(nn_model)
        attr_gs = gs.attribute(input_tensor, baselines=baseline_dist)
        attr_gs = attr_gs.detach().numpy().flatten()

        fig_gs, ax = plt.subplots(figsize=(6, 4))
        sorted_idx = np.argsort(np.abs(attr_gs))[::-1][:10]
        ax.barh(np.array(test_X.columns)[sorted_idx], attr_gs[sorted_idx])
        ax.set_xlabel('Attribution')
        ax.invert_yaxis()
        st.pyplot(fig_gs)




# =========================== Feature Comparison Section ===========================
st.subheader("📊 Top 5 Feature Comparison")
shap_feature_imp = pd.DataFrame({
    'feature': test_X.columns,
    'shap_value': np.abs(shap_values.values[0])
}).sort_values(by='shap_value', ascending=False)
shap_top = shap_feature_imp['feature'].head(5).tolist()

lime_top = [x[0] for x in lime_exp.as_list()[:5]]

comparison_df = pd.DataFrame({
    "Rank": list(range(1, 6)),
    "SHAP Top Features": shap_top,
    "LIME Top Features": lime_top
})
st.table(comparison_df)

# Overlap
overlap = set(shap_top) & set(lime_top)
st.subheader("🔁 Overlap Analysis")
st.write(f"**Overlap features:** `{', '.join(overlap)}`")
st.success(f"✅ {len(overlap)} out of 5 features overlap")

# =========================== User Feedback Section ===========================
st.sidebar.markdown("## 💬 User Feedback")

# Q1: Which model do you trust more?
trusted_model = st.sidebar.radio(
    "1️⃣ Which model do you trust more?",
    ["XGBRegressor", "Neural Network", "Both equally", "Neither"]
)

# Q2 & Q3: Dynamically render based on answer to Q1
if trusted_model == "XGBRegressor":
    understandable = st.sidebar.radio(
        "2️⃣ For XGBRegressor, which explanation is easier to understand?",
        ["SHAP", "LIME", "Both equally", "Neither"]
    )
    trusted_explainer = st.sidebar.radio(
        "3️⃣ For XGBRegressor, which explanation do you trust more?",
        ["SHAP", "LIME", "Both equally", "Neither"]
    )

elif trusted_model == "Neural Network":
    understandable = st.sidebar.radio(
        "2️⃣ For Neural Network, which explanation is easier to understand?",
        ["SHAP", "LIME", "Gradient-based Explanation Methods", "All equally", "None"]
    )
    trusted_explainer = st.sidebar.radio(
        "3️⃣ For Neural Network, which explanation do you trust more?",
        ["SHAP", "LIME", "Gradient-based Explanation Methods", "All equally", "None"]
    )

else:  # Both equally or Neither
    understandable = st.sidebar.radio(
        "2️⃣ Which explanation is easier to understand?",
        ["SHAP", "LIME", "Gradient-based Explanation Methods", "All equally", "None"]
    )
    trusted_explainer = st.sidebar.radio(
        "3️⃣ Which explanation do you trust more?",
        ["SHAP", "LIME", "Gradient-based Explanation Methods", "All equally", "None"]
    )

# Q4: Free-text feedback
feedback = st.sidebar.text_area("🗣️ Any additional comments or suggestions?")

# Submit button
if st.sidebar.button("📩 Submit Feedback"):
    st.sidebar.success("✅ Thank you for your feedback!")



# =========================== Run the dashboard ===========================
# streamlit run combined_xai_dashboard.py