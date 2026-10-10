
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AI SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #808080;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">📩 AI-Based SMS Spam Detection</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Machine Learning Dashboard | '
    'Python for AI Micro-Project</div>',
    unsafe_allow_html=True
)
st.divider()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    data = data.dropna(subset=["label", "message"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str)
    data = data[data["label"].isin(["ham", "spam"])]

    return data.reset_index(drop=True)

try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place the file named "
        "'SMSSpamCollection' in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if data.empty or data["label"].nunique() < 2:
    st.error("The dataset must contain both ham and spam messages.")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "SMS Detector",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "This project uses CountVectorizer and "
    "Multinomial Naive Bayes to classify SMS messages."
)

# ---------------- TRAIN MODEL ----------------
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

@st.cache_resource
def train_model(messages, labels):
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    model.fit(list(messages), list(labels))
    return model

model = train_model(tuple(X_train), tuple(y_train))

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)
recall = recall_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)
f1 = f1_score(
    y_test, y_pred, pos_label="spam", zero_division=0
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.header("📊 Dataset Overview")
    st.write(
        "Explore the SMS dataset and view the overall "
        "performance of the trained machine learning model."
    )

    total_messages = len(data)
    spam_count = int((data["label"] == "spam").sum())
    ham_count = int((data["label"] == "ham").sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("📨 Total Messages", f"{total_messages:,}")
    col2.metric("🚨 Spam Messages", f"{spam_count:,}")
    col3.metric("✅ Legitimate Messages", f"{ham_count:,}")

    st.divider()
    st.subheader("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    st.caption(
        "Metrics are calculated on the held-out test set. "
        "Precision, recall, and F1-score treat spam as the positive class."
    )

    st.divider()
    st.subheader("📊 Message Distribution")

    distribution = pd.Series({
        "Legitimate (Ham)": ham_count,
        "Spam": spam_count
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    distribution.plot(
        kind="bar",
        ax=ax,
        color=["#2EAD68", "#E45756"],
        edgecolor="white"
    )
    ax.set_title("Spam vs Legitimate Messages")
    ax.set_xlabel("Message Category")
    ax.set_ylabel("Number of Messages")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Preview dataset"):
        st.dataframe(data.head(20), use_container_width=True)

# ---------------- SMS DETECTOR ----------------
elif page == "SMS Detector":

    st.header("🔍 Test Your SMS Message")

    st.write(
        "Enter a message below and let the trained AI model "
        "predict whether it resembles spam or legitimate SMS."
    )

    message = st.text_area(
        "SMS message",
        height=150,
        placeholder=(
            "Example: Congratulations! You have won a prize. "
            "Claim it now!"
        )
    )

    if st.button("🔎 Analyze Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")

        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            confidence = float(probabilities[classes.index(prediction)])

            st.subheader("Prediction Result")

            if prediction == "spam":
                st.error("🚨 SPAM MESSAGE DETECTED")
                st.write(
                    "This message resembles spam messages "
                    "in the training dataset."
                )
            else:
                st.success("✅ LEGITIMATE MESSAGE (NOT SPAM)")
                st.write(
                    "This message resembles legitimate messages "
                    "in the training dataset."
                )

            st.metric("Model Probability for Predicted Class",
                      f"{confidence:.2%}")

            st.caption(
                "This probability is the model's estimate, "
                "not a guarantee that the prediction is correct."
            )

            with st.expander("View class probabilities"):
                probability_table = pd.DataFrame({
                    "Category": ["Legitimate (Ham)", "Spam"],
                    "Model Probability": [
                        probabilities[classes.index("ham")],
                        probabilities[classes.index("spam")]
                    ]
                })
                st.dataframe(
                    probability_table.style.format({
                        "Model Probability": "{:.2%}"
                    }),
                    use_container_width=True
                )

# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":

    st.header("🧪 Model Evaluation")
    st.write(
        "Evaluate the classifier using messages that were "
        "held out from model training."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    st.divider()
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Spam"]
    )
    display.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
        values_format="d"
    )
    ax.set_title("Actual vs Predicted Messages")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows represent actual categories; columns represent "
        "predicted categories. Off-diagonal values are errors."
    )

    st.divider()
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=["ham", "spam"],
        target_names=["Legitimate (Ham)", "Spam"],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Classification Report",
        data=report_df.to_csv().encode("utf-8"),
        file_name="classification_report.csv",
        mime="text/csv"
    )

# ---------------- ABOUT ----------------
elif page == "About Project":

    st.header("ℹ️ About the Project")

    st.subheader("Objective")
    st.write(
        "To develop a machine learning system that automatically "
        "classifies SMS messages as spam or legitimate."
    )

    st.subheader("Technologies Used")
    st.markdown("""
    - **Python:** Main programming language
    - **Pandas:** Dataset loading and handling
    - **Scikit-learn:** Machine learning and evaluation
    - **CountVectorizer:** Converts text into numerical features
    - **Multinomial Naive Bayes:** Classifies SMS messages
    - **Streamlit:** Interactive web dashboard
    - **Matplotlib:** Charts and confusion matrix visualization
    """)

    st.subheader("How It Works")
    st.markdown("""
    1. Load and prepare the labelled SMS dataset.
    2. Split the dataset into training and testing sets.
    3. Convert message text into numerical word-count features.
    4. Train the Naive Bayes classifier.
    5. Predict the category of unseen messages.
    6. Evaluate the model using classification metrics.
    """)

    st.warning(
        "This is an educational prototype. Do not rely on its "
        "predictions alone to determine whether a message is safe."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "AI-Based SMS Spam Detection | Python for AI Micro-Project"
)
