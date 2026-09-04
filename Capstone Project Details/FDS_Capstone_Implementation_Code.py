import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime

st.set_page_config(
    page_title="NeuroShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "spam_model.pkl"

st.markdown("""
<style>
    .stApp {
        background-color: #07111f;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1625;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.3px;
    }

    div[data-testid="stMetric"] {
        background-color: #0d1a2b;
        border: 1px solid #1b344d;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
    }

    textarea {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HELPERS
# -----------------------------
def get_value(package, names, default=0):
    for name in names:
        if name in package:
            return package[name]
    return default


def find_dataset():
    candidates = [
        os.path.join("data", "SMSSpamCollection"),
        os.path.join("data", "SMSSpamCollection.txt"),
        "SMSSpamCollection",
        "SMSSpamCollection.txt",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def load_dataset():
    path = find_dataset()

    if not path:
        return None

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
            on_bad_lines="skip",
        )
        return df.dropna(subset=["label", "message"])
    except Exception:
        return None


def message_features(text):
    text = str(text)

    words = re.findall(r"\b\w+\b", text)
    links = re.findall(r"(?:https?://|www\.)\S+", text, flags=re.I)

    suspicious_words = [
        "free", "winner", "won", "prize", "claim",
        "urgent", "cash", "reward", "congratulations",
        "click", "bonus", "limited", "selected",
        "lottery", "voucher", "verify", "account",
        "unsubscribe", "promo", "loan", "guaranteed",
    ]

    lower = text.lower()

    return {
        "words": len(words),
        "characters": len(text),
        "digits": sum(c.isdigit() for c in text),
        "links": len(links),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
        "money": sum(text.count(x) for x in ["₹", "$", "€", "£"]),
        "uppercase": sum(c.isupper() for c in text),
        "suspicious": sorted(
            set(word for word in suspicious_words if word in lower)
        ),
        "links_found": links,
    }


def risk_level(spam_probability):
    if spam_probability >= 0.90:
        return "CRITICAL"
    if spam_probability >= 0.70:
        return "HIGH"
    if spam_probability >= 0.50:
        return "MEDIUM"
    if spam_probability >= 0.25:
        return "LOW"
    return "MINIMAL"


def model_predict(text):
    """
    Uses the saved model exactly as it was trained:
    raw message -> saved vectorizer -> saved classifier.

    Behavioral indicators shown in the UI are explanatory signals;
    they are not secretly added to the model input.
    """
    X = vectorizer.transform([str(text)])
    probabilities = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", [0, 1]))

    if 1 in classes:
        spam_index = classes.index(1)
    else:
        spam_index = int(np.argmax(probabilities))

    if 0 in classes:
        safe_index = classes.index(0)
    else:
        safe_index = 1 - spam_index

    spam_probability = float(probabilities[spam_index])
    safe_probability = float(probabilities[safe_index])

    return spam_probability, safe_probability


def batch_predict(messages):
    X = vectorizer.transform(messages)
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))

    spam_index = classes.index(1) if 1 in classes else int(np.argmax(probabilities))
    safe_index = classes.index(0) if 0 in classes else 1 - spam_index

    spam_prob = probabilities[:, spam_index]
    safe_prob = probabilities[:, safe_index]

    return spam_prob, safe_prob


# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ spam_model.pkl was not found.")
    st.info("Run `python train_model.py` first.")
    st.stop()

try:
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    vectorizer = package["vectorizer"]

except Exception as error:
    st.error("❌ Could not load the trained model.")
    st.code(str(error))
    st.stop()


accuracy = float(get_value(package, ["accuracy"], 0))
precision = float(get_value(package, ["precision"], 0))
recall = float(get_value(package, ["recall"], 0))
f1 = float(get_value(package, ["f1"], 0))

dataset_size = int(get_value(package, ["dataset_size"], 5572))
training_samples = int(get_value(package, ["training_samples"], 4457))
testing_samples = int(get_value(package, ["testing_samples"], 1115))
feature_count = int(get_value(package, ["feature_count"], 0))

confusion = get_value(
    package,
    ["confusion_matrix"],
    [[0, 0], [0, 0]]
)


# -----------------------------
# SESSION STATE
# -----------------------------
if "message" not in st.session_state:
    st.session_state.message = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🛡️ NeuroShield")
    st.caption("AI THREAT INTELLIGENCE")

    st.success("● MODEL ONLINE")
    st.caption("Logistic Regression engine active")

    st.divider()

    st.subheader("⚙️ Model")

    st.write("**Algorithm**")
    st.code("Logistic Regression")

    st.write("**Vectorization**")
    st.code("TF-IDF")

    st.write("**Features**")
    st.code("Text + Behavioral")

    st.divider()

    st.subheader("📡 Dataset")

    st.metric("Messages", f"{dataset_size:,}")
    st.metric("Training", f"{training_samples:,}")
    st.metric("Testing", f"{testing_samples:,}")

    st.divider()

    st.subheader("🎯 Detection threshold")

    threshold = st.slider(
        "Spam probability threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.caption(
        f"≥ {threshold:.0%} is classified as SPAM."
    )


# ============================================================
# HEADER
# ============================================================
st.title("🛡️ NeuroShield AI")
st.subheader("Intelligent Email & SMS Spam Detection")
st.caption(
    "Machine-learning based message analysis with probability scoring "
    "and behavioral indicators."
)

st.success("● SYSTEM OPERATIONAL")

# ============================================================
# PERFORMANCE
# ============================================================
st.header("📈 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")


# ============================================================
# MAIN TABS
# ============================================================
live_tab, batch_tab, dataset_tab, model_tab, history_tab = st.tabs(
    [
        "⚡ Live Analyzer",
        "📦 Batch Scanner",
        "📊 Dataset Explorer",
        "🧠 Model Intelligence",
        "🕘 Recent Scans",
    ]
)


# ============================================================
# LIVE ANALYZER
# ============================================================
with live_tab:

    st.header("🔬 Live Threat Analyzer")
    st.write(
        "Enter any SMS or email message. The trained model will calculate "
        "its spam probability."
    )

    demo1, demo2, demo3, demo4 = st.columns(4)

    with demo1:
        if st.button("🚨 Spam Example", use_container_width=True):
            st.session_state.message = (
                "URGENT! Congratulations! You have WON ₹50,000 CASH! "
                "Claim your FREE prize now by clicking "
                "http://bit.ly/winner123"
            )
            st.rerun()

    with demo2:
        if st.button("🛡️ Safe Example", use_container_width=True):
            st.session_state.message = (
                "Hi, the project meeting has been moved to 3 PM tomorrow. "
                "Please bring the updated presentation."
            )
            st.rerun()

    with demo3:
        if st.button("🔗 Link Example", use_container_width=True):
            st.session_state.message = (
                "Your account requires verification. "
                "Visit https://example.com/verify to continue."
            )
            st.rerun()

    with demo4:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.message = ""
            st.session_state.result = None
            st.rerun()

    st.session_state.message = st.text_area(
        "Message",
        value=st.session_state.message,
        height=180,
        placeholder="Paste an email or SMS message here...",
    )

    if st.button(
        "⚡ ANALYZE MESSAGE",
        type="primary",
        use_container_width=True,
    ):
        if not st.session_state.message.strip():
            st.warning("Please enter a message first.")
        else:
            spam_probability, safe_probability = model_predict(
                st.session_state.message
            )

            features = message_features(st.session_state.message)
            label = (
                "SPAM"
                if spam_probability >= threshold
                else "SAFE"
            )

            result = {
                "label": label,
                "spam": spam_probability,
                "safe": safe_probability,
                "risk": risk_level(spam_probability),
                "features": features,
            }

            st.session_state.result = result

            st.session_state.history.insert(
                0,
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Result": label,
                    "Spam probability": spam_probability,
                    "Message": st.session_state.message[:100],
                }
            )

            st.session_state.history = st.session_state.history[:20]

    if st.session_state.result:

        result = st.session_state.result

        st.divider()
        st.header("🎯 Detection Result")

        if result["label"] == "SPAM":
            st.error(
                f"🚨 SPAM DETECTED\n\n"
                f"Spam probability: {result['spam']:.2%}\n\n"
                f"Risk level: {result['risk']}"
            )
        else:
            st.success(
                f"🛡️ MESSAGE APPEARS SAFE\n\n"
                f"Legitimate probability: {result['safe']:.2%}\n\n"
                f"Risk level: {result['risk']}"
            )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "🚨 Spam",
            f"{result['spam']:.2%}"
        )

        r2.metric(
            "🛡️ Safe",
            f"{result['safe']:.2%}"
        )

        r3.metric(
            "Risk",
            result["risk"]
        )

        r4.metric(
            "Threshold",
            f"{threshold:.0%}"
        )

        st.header("📋 Message Intelligence")

        features = result["features"]

        f1c, f2c, f3c, f4c, f5c, f6c = st.columns(6)

        f1c.metric("📝 Words", features["words"])
        f2c.metric("🔤 Characters", features["characters"])
        f3c.metric("🔢 Digits", features["digits"])
        f4c.metric("🔗 Links", features["links"])
        f5c.metric("❗ Exclamations", features["exclamations"])
        f6c.metric("💰 Money", features["money"])

        st.header("🔍 Detection Signals")

        signals = []

        if features["links"] > 0:
            signals.append(
                f"🔗 {features['links']} link(s) detected"
            )

        if features["digits"] >= 5:
            signals.append(
                f"🔢 High digit frequency ({features['digits']})"
            )

        if features["exclamations"] >= 2:
            signals.append(
                f"❗ Multiple exclamation marks ({features['exclamations']})"
            )

        if features["money"] > 0:
            signals.append(
                "💰 Financial / currency indicator detected"
            )

        if features["uppercase"] >= 10:
            signals.append(
                "🔠 High uppercase character usage"
            )

        if features["suspicious"]:
            signals.append(
                "⚠️ Suspicious vocabulary: "
                + ", ".join(features["suspicious"])
            )

        if signals:
            for signal in signals:
                st.warning(signal)
        else:
            st.success("No obvious behavioral warning signals detected.")

        st.header("📊 Probability Analysis")

        st.write(
            f"🛡️ Legitimate probability — "
            f"**{result['safe']:.2%}**"
        )
        st.progress(result["safe"])

        st.write(
            f"🚨 Spam probability — "
            f"**{result['spam']:.2%}**"
        )
        st.progress(result["spam"])

        if features["links_found"]:
            st.header("🔗 Link Intelligence")
            for link in features["links_found"]:
                st.code(link)


# ============================================================
# BATCH SCANNER
# ============================================================
with batch_tab:

    st.header("📦 Batch Scanner")

    st.write(
        "Upload a CSV containing many messages and classify them together. "
        "This is the practical use of the complete dataset/model."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="The CSV should contain a message/text column.",
    )

    if uploaded_file is None:
        st.info(
            "No CSV uploaded yet. The Live Analyzer works without a CSV."
        )
    else:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f"Loaded {len(batch_df):,} rows."
            )

            candidates = [
                column for column in batch_df.columns
                if str(column).lower() in [
                    "message",
                    "text",
                    "sms",
                    "email",
                    "content",
                    "body",
                ]
            ]

            default_index = (
                list(batch_df.columns).index(candidates[0])
                if candidates
                else 0
            )

            message_column = st.selectbox(
                "Choose the message column",
                batch_df.columns,
                index=default_index,
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "🚀 SCAN ALL MESSAGES",
                type="primary",
                use_container_width=True,
            ):

                messages = (
                    batch_df[message_column]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )

                with st.spinner(
                    f"Analyzing {len(messages):,} messages..."
                ):
                    spam_prob, safe_prob = batch_predict(messages)

                results = batch_df.copy()

                results["Prediction"] = np.where(
                    spam_prob >= threshold,
                    "SPAM",
                    "SAFE",
                )

                results["Spam Probability"] = spam_prob
                results["Safe Probability"] = safe_prob

                total = len(results)
                spam_total = int(
                    (results["Prediction"] == "SPAM").sum()
                )
                safe_total = total - spam_total

                b1, b2, b3, b4 = st.columns(4)

                b1.metric("Scanned", f"{total:,}")
                b2.metric("Spam", f"{spam_total:,}")
                b3.metric("Safe", f"{safe_total:,}")
                b4.metric(
                    "Spam Rate",
                    f"{spam_total / total:.1%}"
                    if total else "0%",
                )

                st.dataframe(
                    results,
                    use_container_width=True,
                    height=450,
                    hide_index=True,
                )

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ DOWNLOAD RESULTS CSV",
                    data=csv_output,
                    file_name="neuroshield_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as error:
            st.error("Could not process the CSV.")
            st.code(str(error))


# ============================================================
# DATASET EXPLORER
# ============================================================
with dataset_tab:

    st.header("📊 Dataset Explorer")

    data = load_dataset()

    if data is None:
        st.warning(
            "The dataset file was not found in the data folder."
        )
        st.write(
            "Expected file: `data/SMSSpamCollection`"
        )
    else:

        data["label"] = data["label"].astype(str).str.lower()

        total = len(data)
        spam_total = int((data["label"] == "spam").sum())
        safe_total = total - spam_total

        d1, d2, d3, d4 = st.columns(4)

        d1.metric("Total Messages", f"{total:,}")
        d2.metric("Spam", f"{spam_total:,}")
        d3.metric("Safe / Ham", f"{safe_total:,}")
        d4.metric(
            "Spam Ratio",
            f"{spam_total / total:.1%}"
            if total else "0%",
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Class Distribution")

            distribution = pd.DataFrame(
                {
                    "Messages": [
                        safe_total,
                        spam_total,
                    ]
                },
                index=["Safe / Ham", "Spam"],
            )

            st.bar_chart(distribution)

        with right:
            st.subheader("Message Length")

            lengths = data["message"].astype(str).str.len()

            groups = pd.cut(
                lengths,
                bins=[
                    -1,
                    40,
                    80,
                    120,
                    200,
                    500,
                    np.inf,
                ],
                labels=[
                    "0-40",
                    "41-80",
                    "81-120",
                    "121-200",
                    "201-500",
                    "500+",
                ],
            )

            st.bar_chart(
                groups.value_counts().sort_index()
            )

        st.subheader("🔎 Search Dataset")

        search_text = st.text_input(
            "Search message text",
            placeholder="Try: free, meeting, winner, claim",
        )

        if search_text:
            filtered = data[
                data["message"]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False,
                )
            ]
        else:
            filtered = data.sample(
                min(25, len(data)),
                random_state=42,
            )

        st.caption(
            f"Showing {len(filtered):,} records"
        )

        st.dataframe(
            filtered,
            use_container_width=True,
            height=450,
            hide_index=True,
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================
with model_tab:

    st.header("🧠 Model Intelligence")

    st.subheader("How NeuroShield AI Works")

    step1, step2 = st.columns(2)

    with step1:
        st.info(
            "01 — Text Processing\n\n"
            "Incoming SMS/email text is passed into the trained "
            "text-processing pipeline."
        )

        st.info(
            "02 — TF-IDF Feature Extraction\n\n"
            "Words and phrases are converted into numerical features "
            "that the classifier can understand."
        )

    with step2:
        st.info(
            "03 — Logistic Regression\n\n"
            "The trained classifier estimates the probability of the "
            "message belonging to the spam class."
        )

        st.info(
            "04 — Threat Decision\n\n"
            "The spam probability is compared with the threshold "
            "selected in the sidebar."
        )

    st.subheader("⚡ Detection Pipeline")

    st.code(
        "MESSAGE\n"
        "   ↓\n"
        "TF-IDF FEATURE EXTRACTION\n"
        "   ↓\n"
        "LOGISTIC REGRESSION\n"
        "   ↓\n"
        "SPAM PROBABILITY\n"
        "   ↓\n"
        "THRESHOLD\n"
        "   ↓\n"
        "SPAM / SAFE"
    )

    st.subheader("🎯 Confusion Matrix")

    try:
        tn = int(confusion[0][0])
        fp = int(confusion[0][1])
        fn = int(confusion[1][0])
        tp = int(confusion[1][1])
    except Exception:
        tn = fp = fn = tp = 0

    cm1, cm2, cm3, cm4 = st.columns(4)

    cm1.metric("True Negative", tn)
    cm2.metric("False Positive", fp)
    cm3.metric("False Negative", fn)
    cm4.metric("True Positive", tp)

    st.subheader("📈 Model Benchmark")

    benchmark = pd.DataFrame(
        {
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
            ]
        },
        index=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
    )

    st.bar_chart(benchmark)

    st.subheader("⚙️ Model Configuration")

    configuration = pd.DataFrame(
        {
            "Property": [
                "Classifier",
                "Vectorizer",
                "Training samples",
                "Testing samples",
                "Dataset size",
                "Feature count",
            ],
            "Value": [
                "Logistic Regression",
                "TF-IDF",
                f"{training_samples:,}",
                f"{testing_samples:,}",
                f"{dataset_size:,}",
                f"{feature_count:,}",
            ],
        }
    )

    st.dataframe(
        configuration,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# HISTORY
# ============================================================
with history_tab:

    st.header("🕘 Recent Scans")

    if not st.session_state.history:
        st.info(
            "No scans yet. Analyze a message in the Live Analyzer."
        )
    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        history_df["Spam probability"] = (
            history_df["Spam probability"]
            .map(lambda x: f"{x:.1%}")
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "🧹 CLEAR HISTORY",
            use_container_width=True,
        ):
            st.session_state.history = []
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "NEUROSHIELD AI • Intelligent Spam Detection • "
    "Logistic Regression • TF-IDF • Hackathon Prototype"
)
