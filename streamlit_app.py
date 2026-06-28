import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Enterprise RAG Platform",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Configuration")

    top_k = st.slider("Top K Results", 1, 10, 5)
    detect_hall = st.checkbox("Detect Hallucinations", value=True)

    st.divider()

    st.markdown("### API Status")

    try:
        r = requests.get(
            "http://localhost:8000/health",
            timeout=2
        )
        st.success("API Online") if r.ok else st.error("API Offline")
    except:
        st.error("API Offline")

# Main Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])

# ==========================
# Chat Tab
# ==========================
with tab1:

    st.title("Enterprise RAG Platform")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input(
        "Ask a question about your documents..."
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Retrieving and generating..."):

                response = requests.post(
                    "http://localhost:8000/query",
                    json={
                        "question": question,
                        "top_k": top_k,
                        "detect_hallucinations": detect_hall
                    }
                )

                data = response.json()

                st.markdown(data["answer"])

                st.caption(
                    f"Latency: {data['latency_ms']:.0f} ms"
                )

                if data.get("hallucination_report"):

                    h = data["hallucination_report"]

                    rate = h["hallucination_rate"]

                    color = (
                        "green"
                        if rate < 0.2
                        else "orange"
                        if rate < 0.5
                        else "red"
                    )

                    st.markdown(
                        f"Hallucination Rate: :{color}[{rate:.1%}]"
                    )

                with st.expander("Retrieved Sources"):

                    for i, src in enumerate(
                        data.get("sources", [])
                    ):

                        st.markdown(
                            f"**Source {i+1}** "
                            f"(relevance: {src.get('relevance_score', 0):.2f})"
                        )

                        st.text(src["text"][:300] + "...")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"]
                    }
                )

# ==========================
# Evaluation Dashboard
# ==========================
with tab2:

    st.title("Evaluation Dashboard")

    st.info(
        "Run evaluation/ragas_suite.py "
        "to generate results, then load here."
    )

    uploaded = st.file_uploader(
        "Upload eval_results.csv",
        type="csv"
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        col1, col2, col3, col4 = st.columns(4)

        for metric, col in zip(
            [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall"
            ],
            [col1, col2, col3, col4]
        ):

            if metric in df.columns:

                col.metric(
                    metric.replace("_", " ").title(),
                    f"{df[metric].mean():.3f}"
                )

        if "faithfulness" in df.columns:

            fig = px.histogram(
                df,
                x="faithfulness",
                title="Faithfulness Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )