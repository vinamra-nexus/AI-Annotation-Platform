# ==================================================

# IMPORT LIBRARIES

# ==================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==================================================

# LOAD TRAINED MACHINE LEARNING MODEL

# ==================================================

# Load trained sentiment classification model

model = joblib.load("sentiment_model.pkl")

# Load TF-IDF vectorizer

vectorizer = joblib.load("vectorizer.pkl")

# ==================================================

# AUTHENTICATION CONFIGURATION

# ==================================================

USERNAME = "admin"
PASSWORD = "admin123"

# ==================================================

# APPLICATION HEADER

# ==================================================

st.title("🧠 AI Data Annotation Tool - PRO MODE")

# ==================================================

# LOGIN SECTION

# ==================================================

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")

password = st.sidebar.text_input(
"Password",
type="password"
)

if username != USERNAME or password != PASSWORD:
st.warning("Please login")
st.stop()

st.sidebar.success(f"✅ Logged in as {username}")

# ==================================================

# DATASET UPLOAD SECTION

# ==================================================

uploaded_file = st.file_uploader(
"Upload CSV file",
type=["csv"]
)

# ==================================================

# MAIN APPLICATION

# ==================================================

if uploaded_file is not None:

```
# ----------------------------------------------
# LOAD DATASET
# ----------------------------------------------

df = pd.read_csv(uploaded_file)

# ----------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = 0

if "labels" not in st.session_state:
    st.session_state.labels = []

# ----------------------------------------------
# CURRENT RECORD SELECTION
# ----------------------------------------------

row = df.iloc[st.session_state.index]
text = row["text"]

# ----------------------------------------------
# ANNOTATION PROGRESS TRACKER
# ----------------------------------------------

total_records = len(df)
completed = len(st.session_state.labels)

st.progress(completed / total_records)

st.write(
    f"{completed}/{total_records} records annotated"
)

# ==================================================
# MACHINE LEARNING PREDICTION FUNCTION
# ==================================================

def predict_label(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    return prediction[0]

# ==================================================
# AI LABEL PREDICTION & CONFIDENCE SCORE
# ==================================================

text_vector = vectorizer.transform([text])

ai_label = model.predict(text_vector)[0]

confidence = max(
    model.predict_proba(text_vector)[0]
) * 100

# ==================================================
# ANNOTATION INTERFACE
# ==================================================

st.write("### 📄 Text to Label")
st.info(text)

st.write("### 🤖 AI Suggested Label")
st.success(ai_label)

st.info(
    f"Confidence: {confidence:.1f}%"
)

# ==================================================
# BULK AUTO ANNOTATION
# ==================================================

if st.button("🤖 Auto Annotate All"):

    st.session_state.labels = []

    for _, row in df.iterrows():

        auto_label = predict_label(row["text"])

        st.session_state.labels.append({
            "text": row["text"],
            "ai_suggestion": auto_label,
            "final_label": auto_label
        })

    st.success(
        "All records annotated automatically!"
    )

    st.rerun()

# ==================================================
# MANUAL LABEL SELECTION
# ==================================================

label = st.selectbox(
    "Choose Final Label",
    ["Positive", "Negative", "Neutral"]
)

# ==================================================
# SAVE ANNOTATION
# ==================================================

if st.button("💾 Save Annotation"):

    current_text = text

    already_saved = any(
        item["text"] == current_text
        for item in st.session_state.labels
    )

    if already_saved:

        st.warning(
            "This row is already labeled!"
        )

    else:

        st.session_state.labels.append({
            "text": text,
            "ai_suggestion": ai_label,
            "final_label": label
        })

        st.success("Saved!")

        if st.session_state.index < len(df) - 1:
            st.session_state.index += 1

        st.rerun()

# ==================================================
# RECORD NAVIGATION
# ==================================================

col1, col2 = st.columns(2)

if col1.button("⬅️ Previous"):

    if st.session_state.index > 0:

        st.session_state.index -= 1

        st.rerun()

if col2.button("Next ➡️"):

    if st.session_state.index < len(df) - 1:

        st.session_state.index += 1

        st.rerun()

# ==================================================
# ANNOTATION MANAGEMENT & EXPORT
# ==================================================

if len(st.session_state.labels) > 0:

    st.write("## 📦 Exported Annotations")

    result_df = pd.DataFrame(
        st.session_state.labels
    )

    # ----------------------------------------------
    # SEARCH & FILTER ANNOTATIONS
    # ----------------------------------------------

    search = st.text_input(
        "🔍 Search Annotations"
    )

    selected_label = st.selectbox(
        "Filter by Label",
        ["All", "Positive", "Negative", "Neutral"]
    )

    filtered_df = result_df.copy()

    if selected_label != "All":

        filtered_df = filtered_df[
            filtered_df["final_label"]
            == selected_label
        ]

    if search:

        filtered_df = filtered_df[
            filtered_df["text"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(filtered_df)

    # ----------------------------------------------
    # EDIT EXISTING ANNOTATIONS
    # ----------------------------------------------

    st.write("### ✏️ Edit Saved Annotations")

    for i, item in enumerate(
        st.session_state.labels
    ):

        with st.expander(
            f"Annotation {i+1}"
        ):

            st.write(item["text"])

            new_label = st.selectbox(
                "Edit Label",
                [
                    "Positive",
                    "Negative",
                    "Neutral"
                ],
                index=[
                    "Positive",
                    "Negative",
                    "Neutral"
                ].index(
                    item["final_label"]
                ),
                key=f"edit_{i}"
            )

            if st.button(
                "Update",
                key=f"update_{i}"
            ):

                st.session_state.labels[i][
                    "final_label"
                ] = new_label

                st.rerun()

    # ----------------------------------------------
    # EXPORT ANNOTATED DATASET
    # ----------------------------------------------

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Annotated CSV",
        csv,
        "annotations_pro.csv",
        "text/csv"
    )

    st.write(
        f"Total Annotations: "
        f"{len(st.session_state.labels)}"
    )

    # ==================================================
    # LABEL DISTRIBUTION STATISTICS
    # ==================================================

    st.write(
        "## 📊 Annotation Statistics"
    )

    positive = sum(
        item["final_label"] == "Positive"
        for item in st.session_state.labels
    )

    negative = sum(
        item["final_label"] == "Negative"
        for item in st.session_state.labels
    )

    neutral = sum(
        item["final_label"] == "Neutral"
        for item in st.session_state.labels
    )

    st.write(f"✅ Positive: {positive}")
    st.write(f"❌ Negative: {negative}")
    st.write(f"➖ Neutral: {neutral}")

    # ==================================================
    # AI PERFORMANCE EVALUATION
    # ==================================================

    matches = 0

    for item in st.session_state.labels:

        if (
            item["ai_suggestion"]
            == item["final_label"]
        ):
            matches += 1

    accuracy = (
        matches
        / len(st.session_state.labels)
        * 100
    )

    st.write("## 🤖 AI Performance")

    st.metric(
        "AI Accuracy",
        f"{accuracy:.1f}%"
    )

    # ==================================================
    # HUMAN REVIEW METRICS
    # ==================================================

    accepted = 0
    corrected = 0

    for item in st.session_state.labels:

        if (
            item["ai_suggestion"]
            == item["final_label"]
        ):
            accepted += 1
        else:
            corrected += 1

    correction_rate = (
        corrected
        / len(st.session_state.labels)
        * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Accepted",
        accepted
    )

    col2.metric(
        "AI Corrected",
        corrected
    )

    col3.metric(
        "Correction Rate",
        f"{correction_rate:.1f}%"
    )

    # ==================================================
    # ANALYTICS DASHBOARD
    # ==================================================

    st.write(
        "## 📊 Analytics Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total",
        len(result_df)
    )

    col2.metric(
        "Positive",
        len(
            result_df[
                result_df["final_label"]
                == "Positive"
            ]
        )
    )

    col3.metric(
        "Negative",
        len(
            result_df[
                result_df["final_label"]
                == "Negative"
            ]
        )
    )

    col4.metric(
        "Neutral",
        len(
            result_df[
                result_df["final_label"]
                == "Neutral"
            ]
        )
    )

    st.metric(
        "Total Annotations",
        len(result_df)
    )

    st.bar_chart(
        result_df[
            "final_label"
        ].value_counts()
    )

    # ==================================================
    # DONUT CHART VISUALIZATION
    # ==================================================

    counts = result_df[
        "final_label"
    ].value_counts()

    fig, ax = plt.subplots()

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 2,
            "width": 0.5
        }
    )

    st.pyplot(fig)

    # ==================================================
    # ANALYTICS REPORT EXPORT
    # ==================================================

    st.write(
        "## 📥 Download Analytics Report"
    )

    analytics_df = pd.DataFrame({

        "Metric": [
            "Total",
            "Positive",
            "Negative",
            "Neutral"
        ],

        "Count": [
            len(result_df),

            len(
                result_df[
                    result_df["final_label"]
                    == "Positive"
                ]
            ),

            len(
                result_df[
                    result_df["final_label"]
                    == "Negative"
                ]
            ),

            len(
                result_df[
                    result_df["final_label"]
                    == "Neutral"
                ]
            )
        ]
    })

    st.download_button(
        "Download Analytics Report",
        analytics_df.to_csv(
            index=False
        ),
        "analytics_report.csv",
        "text/csv"
    )

    # ==================================================
    # SESSION SUMMARY
    # ==================================================

    st.write("## 📋 Session Summary")

    total = len(df)

    annotated = len(
        st.session_state.labels
    )

    remaining = total - annotated

    most_common = (
        result_df["final_label"]
        .value_counts()
        .idxmax()
    )

    st.write(
        f"📄 Total Records: {total}"
    )

    st.write(
        f"✅ Annotated: {annotated}"
    )

    st.write(
        f"⏳ Remaining: {remaining}"
    )

    st.write(
        f"🏆 Most Common Label: {most_common}"
    )
```
