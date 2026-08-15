import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

from training_scripts.BreastCancerDataPreprocessor import BreastCancerDataPreprocessor
from training_scripts.model_utils import predict_with_best_model


st.set_page_config(
    page_title="Breast Cancer Diagnostic Predictor",
    page_icon="🩺",
    layout="wide"
)

# 30 continuous features of Breast Cancer Wisconsin dataset
FEATURE_COLUMNS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

DEFAULT_SAMPLE_1 = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

DEFAULT_SAMPLE_2 = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

# --- NATIVE INTERACTIVE WALKTHROUGH DIALOG ---
@st.dialog("👋 App Walkthrough Tutorial")
def show_walkthrough():
    if "tour_step" not in st.session_state:
        st.session_state.tour_step = 1

    step = st.session_state.tour_step
    st.progress(step / 4)

    if step == 1:
        st.markdown("### Step 1: Choose Your Input Method")
        st.markdown(
            "Use the radio buttons at the top of the main area to select how you want to feed data into the model:\n"
            "* **Upload CSV:** Drag and drop or browse a `.csv` file with feature columns.\n"
            "* **Manual Tabular Input:** Type feature values directly into an interactive grid."
        )
    elif step == 2:
        st.markdown("### Step 2: Configure Model Options (Sidebar)")
        st.markdown(
            "Expand or view the **⚙️ Model Configuration** sidebar on the left:\n"
            "* **Metric Selection:** Choose which performance metric determines the best model for inference.\n"
            "* **Model Filter:** Restrict inference to a specific algorithm or leave it as `All Models (Global Best)`."
        )
    elif step == 3:
        st.markdown("### Step 3: Prepare & Inspect Input Data")
        st.markdown(
            "Depending on your chosen input mode:\n"
            "* **CSV Mode:** Preview loaded rows before running predictions.\n"
            "* **Manual Mode:** Toggle **'Fill Default Baseline Values'** or add/delete custom rows."
        )
    elif step == 4:
        st.markdown("### Step 4: Run Inference & Analyze Results")
        st.markdown(
            "Click **🚀 Run Diagnosis Predictions** to process your data:\n"
            "* View predicted diagnoses mapped to **M** (Malignant) or **B** (Benign).\n"
            "* Review Confusion Matrix & Classification Report if ground truth (`diagnosis`) is provided."
        )

    col_back, col_next, col_skip = st.columns([1, 1, 1])

    with col_back:
        if step > 1:
            if st.button("← Back", use_container_width=True):
                st.session_state.tour_step -= 1
                st.rerun()

    with col_next:
        if step < 4:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.tour_step += 1
                st.rerun()
        else:
            if st.button("Get Started 🎉", type="primary", use_container_width=True):
                st.session_state.show_tour = False
                st.session_state.tour_step = 1
                st.rerun()

    with col_skip:
        if st.button("Skip Tutorial", use_container_width=True):
            st.session_state.show_tour = False
            st.session_state.tour_step = 1
            st.rerun()


if "show_tour" not in st.session_state:
    st.session_state.show_tour = True

if st.session_state.show_tour:
    show_walkthrough()

# Header & Subtitle
st.title("🩺 Breast Cancer Wisconsin Diagnostic Predictor")
st.markdown("Upload test CSVs or manually input feature values to generate predictions using the trained model pipeline.")

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")

if st.sidebar.button("❓ Replay Walkthrough Tutorial"):
    st.session_state.show_tour = True
    st.session_state.tour_step = 1
    st.rerun()

selected_metric = st.sidebar.selectbox(
    "Metric to Select Best Model",
    ["accuracy", "f1_score", "auc_score", "mcc_score", "precision", "recall"]
)
model_filter = st.sidebar.selectbox(
    "Model Filter",
    [None, "Random_Forest", "Decision_Tree", "Logistic_Regression", "KNN", "Naive_Bayes"],
    format_func=lambda x: "All Models (Global Best)" if x is None else x
)

st.write("---")

# Section i) Input Method Selection
input_type = st.radio(
    "Select Input Method:",
    options=["Upload CSV", "Manual Tabular Input"],
    index=0,
    horizontal=True
)

raw_input_df = None

# Option 1: CSV File Upload
if input_type == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"], help="Drag and drop your dataset here")
    if uploaded_file is not None:
        raw_input_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded CSV with **{raw_input_df.shape[0]}** rows and **{raw_input_df.shape[1]}** columns.")
        st.subheader("📋 Dataset Preview")
        st.dataframe(raw_input_df.head(), use_container_width=True)

# Option 2: Interactive Data Entry Table
else:
    st.subheader("✍️ Interactive Feature Data Entry")
    col1, _ = st.columns([1, 4])

    with col1:
        load_defaults = st.toggle("Fill Default Baseline Values", value=False)

    if "prev_defaults" not in st.session_state:
        st.session_state.prev_defaults = load_defaults
        st.session_state.editor_key_counter = 0

    if load_defaults != st.session_state.prev_defaults:
        st.session_state.prev_defaults = load_defaults
        st.session_state.editor_key_counter += 1

    if load_defaults:
        data = [
            dict(zip(["diagnosis"] + FEATURE_COLUMNS, ["M"] + DEFAULT_SAMPLE_1)),
            dict(zip(["diagnosis"] + FEATURE_COLUMNS, ["B"] + DEFAULT_SAMPLE_2))
        ]
        initial_df = pd.DataFrame(data)
    else:
        empty_row = dict(zip(["diagnosis"] + FEATURE_COLUMNS, [None] + [0.0] * len(FEATURE_COLUMNS)))
        initial_df = pd.DataFrame([empty_row])

    st.markdown("Add, remove, or edit any row below (**Diagnosis** is optional; select `None` to clear):")

    column_config = {
        "diagnosis": st.column_config.SelectboxColumn(
            "diagnosis (Optional)",
            help="Actual diagnosis label if available. Select None to remove the value.",
            options=[None, "M", "B"],
            required=False
        )
    }

    editor_key = f"data_editor_{st.session_state.editor_key_counter}"

    edited_df = st.data_editor(
        initial_df,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key
    )

    raw_input_df = edited_df.replace({"": np.nan, "None": np.nan})

st.write("---")

# Section ii & iii) Prediction Execution & Display
if st.button("🚀 Run Diagnosis Predictions", type="primary", use_container_width=True):
    if raw_input_df is None or raw_input_df.empty:
        st.error("Please provide valid input data before predicting.")
    else:
        with st.spinner("Preprocessing input features and predicting..."):
            try:
                y_val = None

                if "diagnosis" in raw_input_df.columns:
                    if raw_input_df["diagnosis"].isna().all():
                        raw_input_df = raw_input_df.drop(columns=["diagnosis"])
                    else:
                        y_val = raw_input_df["diagnosis"]
                        raw_input_df = raw_input_df.drop(columns=["diagnosis"])

                processor = BreastCancerDataPreprocessor()
                X_val, _ = processor.preprocess_from_dataframe(raw_input_df, is_training=False)

                results_json = predict_with_best_model(
                    test_x_df=X_val,
                    test_y=y_val if (y_val is not None and not y_val.isna().any()) else None,
                    model_name=model_filter,
                    metric_key=selected_metric,
                    target_name="predicted_diagnosis",
                    save_output=False,
                    raw_features_df=raw_input_df
                )

                predictions_dict = results_json["predictions"]
                output_df = pd.DataFrame.from_dict(predictions_dict, orient='index')

                st.subheader(f"🎯 Inference Results (Model Used: {results_json['model_used']})")

                # --- DISPLAY METRICS & EVALUATION REPORT IF GROUND TRUTH IS PRESENT ---
                if "actual_diagnosis" in output_df.columns and output_df["actual_diagnosis"].notna().any():
                    valid_mask = output_df["actual_diagnosis"].notna()
                    y_true = output_df.loc[valid_mask, "actual_diagnosis"].astype(str)
                    y_pred = output_df.loc[valid_mask, "predicted_diagnosis"].astype(str)

                    st.markdown("## 📊 Evaluation Metrics & Diagnostics")

                    # Display Summary Metric Cards
                    if results_json.get("test_evaluation_metrics"):
                        metrics = results_json["test_evaluation_metrics"]
                        m_cols = st.columns(6)
                        for i, (m_k, m_v) in enumerate(metrics.items()):
                            val_str = f"{m_v:.4f}" if isinstance(m_v, (float, int)) and m_v is not None else "N/A"
                            m_cols[i % 6].metric(m_k.upper(), val_str)

                    st.markdown("---")
                    col_cm, col_cr = st.columns([1, 1])

                    labels = ["B", "M"]

                    # Interactive Confusion Matrix Plot
                    with col_cm:
                        st.markdown("### 🧩 Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred, labels=labels)

                        fig = px.imshow(
                            cm,
                            x=labels,
                            y=labels,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted Diagnosis", y="Actual Diagnosis", color="Count")
                        )
                        fig.update_layout(
                            xaxis_title="Predicted Label",
                            yaxis_title="True Label",
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Text Classification Report
                    with col_cr:
                        st.markdown("### 📝 Classification Report")
                        report = classification_report(y_true, y_pred, labels=labels, output_dict=False)
                        st.code(report, language="text")

                # --- DISPLAY PREDICTIONS TABLE ---
                pred_cols = [c for c in ["predicted_diagnosis", "actual_diagnosis"] if c in output_df.columns]
                other_cols = [c for c in output_df.columns if c not in pred_cols]
                final_df = output_df[pred_cols + other_cols]

                def style_predictions(row):
                    styles = [''] * len(row)
                    has_actual = "actual_diagnosis" in row.index and pd.notna(row["actual_diagnosis"])

                    if has_actual and str(row["predicted_diagnosis"]) != str(row["actual_diagnosis"]):
                        return ['background-color: #8c1d1d; color: white; font-weight: bold;'] * len(row)

                    for idx, col_name in enumerate(row.index):
                        if col_name == "predicted_diagnosis":
                            styles[idx] = 'background-color: #1b5e20; color: white; font-weight: bold;'
                        elif col_name == "actual_diagnosis":
                            styles[idx] = 'background-color: #0d47a1; color: white; font-weight: bold;'

                    return styles

                styled_df = final_df.style.apply(style_predictions, axis=1)

                st.subheader("📋 Output Predictions Table")
                st.dataframe(styled_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")