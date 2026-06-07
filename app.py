import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🧠 AI Data Annotation Tool - PRO MODE")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # session state setup
    if "index" not in st.session_state:
        st.session_state.index = 0

    if "labels" not in st.session_state:
        st.session_state.labels = []

    row = df.iloc[st.session_state.index]
    text = row["text"]

    st.progress(
    (st.session_state.index + 1) / len(df)
    )

    st.write(
        f"Row {st.session_state.index + 1} of {len(df)}"
    )

    # -------------------------
    # 🧠 SIMPLE AI SUGGESTION LOGIC
    # -------------------------
    def predict_label(text):

        text = text.lower()

        positive_words = [
            "love",
            "great",
            "excellent",
            "amazing",
            "fantastic",
            "worth",
            "happy",
            "recommended",
            "superb"
        ]

        negative_words = [
            "worst",
            "terrible",
            "waste",
            "regret",
            "disappointed",
            "awful",
            "horrible",
            "bad",
            "very bad",
            "never buy"
        ]

        if "not too bad" in text or "not that bad" in text or "not bad" in text:
            return "Neutral"

        positive_score = sum(word in text for word in positive_words)
        negative_score = sum(word in text for word in negative_words)

        if positive_score > negative_score:
            return "Positive"

        elif negative_score > positive_score:
            return "Negative"

        else:
            return "Neutral"

    ai_label = predict_label(text)

    # -------------------------
    # 🎨 UI CARD
    # -------------------------
    st.write("### 📄 Text to Label")
    st.info(text)

    st.write("### 🤖 AI Suggested Label")
    st.success(ai_label)

    # manual selection
    label = st.selectbox("Choose Final Label", ["Positive", "Negative", "Neutral"])

    # -------------
    # SAVE
    # -------------
    if st.button("💾 Save Annotation"):

        current_text = text

        already_saved = any(
            item["text"] == current_text
            for item in st.session_state.labels
        )

        if already_saved:
            st.warning("This row is already labeled!")

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

    # navigation
    col1, col2 = st.columns(2)

    if col1.button("⬅️ Previous"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
            st.rerun()

    if col2.button("Next ➡️"):
        if st.session_state.index < len(df) - 1:
            st.session_state.index += 1
            st.rerun()

    # -------------------------
    # 📊 EXPORT SECTION
    # -------------------------
    if len(st.session_state.labels) > 0:
        st.write("## 📦 Exported Annotations")

        result_df = pd.DataFrame(st.session_state.labels)
        st.dataframe(result_df)

        st.write("### ✏️ Edit Saved Annotations")

        for i, item in enumerate(st.session_state.labels):

            with st.expander(f"Annotation {i+1}"):

                st.write(item["text"])

                new_label = st.selectbox(
                    "Edit Label",
                    ["Positive", "Negative", "Neutral"],
                    index=["Positive", "Negative", "Neutral"].index(
                        item["final_label"]
                    ),
                    key=f"edit_{i}"
                )

                if st.button("Update", key=f"update_{i}"):

                    st.session_state.labels[i]["final_label"] = new_label

                    st.rerun()

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Annotated CSV",
            csv,
            "annotations_pro.csv",
            "text/csv"
        )

        st.write(
            f"Total Annotations: {len(st.session_state.labels)}"
        )    

        st.write("## 📊 Annotation Statistics")

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

        st.write("## 📊 Analytics Dashboard")

        result_df = pd.DataFrame(st.session_state.labels)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total", len(result_df))

        col2.metric(
            "Positive",
            len(result_df[result_df["final_label"] == "Positive"])
        )

        col3.metric(
            "Negative",
            len(result_df[result_df["final_label"] == "Negative"])
        )

        col4.metric(
            "Neutral",
            len(result_df[result_df["final_label"] == "Neutral"])
        )

        st.metric("Total Annotations", len(result_df))

        st.bar_chart(
            result_df["final_label"].value_counts()
        )