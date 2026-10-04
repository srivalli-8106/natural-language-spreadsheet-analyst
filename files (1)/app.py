import os

import streamlit as st

from analyst import EVAL_QUESTIONS, LLM, SpreadsheetAgent

st.set_page_config(page_title="NL Spreadsheet Analyst", page_icon="📊", layout="wide")
st.title("📊 Natural Language Spreadsheet Analyst")
st.caption("Upload a messy CSV, ask a question in English, get an insight + table + chart.")

with st.sidebar:
    st.header("Setup")
    key = st.text_input("Groq / OpenAI API key", type="password",
                        value=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                        help="Free Groq key: console.groq.com (starts with gsk_)")
    max_rows = st.number_input("Max rows to load (0 = all)", min_value=0, value=200_000, step=50_000)
    file = st.file_uploader("CSV file", type=["csv", "txt"])

if not file:
    st.info("Upload a CSV in the sidebar to begin (e.g. a sample of the NYC 311 dataset).")
    st.stop()
if not key:
    st.warning("Enter an API key in the sidebar.")
    st.stop()

sig = (file.name, file.size, key, max_rows)
if st.session_state.get("sig") != sig:
    try:
        with st.spinner("Loading and cleaning data..."):
            st.session_state.agent = SpreadsheetAgent(
                file.getvalue(), llm=LLM(api_key=key), max_rows=max_rows or None)
            st.session_state.sig = sig
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load this file: {e}")
        st.stop()

agent: SpreadsheetAgent = st.session_state.agent
rep = agent.report
with st.expander(f"Data loaded: {rep['rows_out']:,} rows x {rep['cols_out']} columns "
                 f"(cleaning report)"):
    for n in rep["notes"] or ["No cleaning issues found."]:
        st.write("•", n)
    st.write("Null counts after cleaning:", rep["null_counts"] or "none")
    st.dataframe(agent.df.head(20), use_container_width=True)

st.subheader("Ask a question")
cols = st.columns(len(EVAL_QUESTIONS))
for i, q in enumerate(EVAL_QUESTIONS):
    if cols[i].button(q, key=f"ex{i}", use_container_width=True):
        st.session_state.q = q
question = st.text_input("Your question", key="q", placeholder="e.g. Which borough has the most noise complaints?")

if question:
    with st.spinner("Thinking..."):
        ans = agent.ask(question)
    if ans.error:
        st.error(ans.insight)
        st.caption(f"Details: {ans.error}")
    else:
        st.success(f"💡 {ans.insight}")
        left, right = st.columns([2, 3])
        with left:
            st.markdown("**Table**")
            st.dataframe(ans.table, use_container_width=True)
            st.download_button("Download table (CSV)", ans.table.to_csv(index=False),
                               "result.csv", "text/csv")
        with right:
            st.markdown("**Chart**")
            if ans.figure is not None:
                st.plotly_chart(ans.figure, use_container_width=True)
            else:
                st.info("No chart could be drawn for this result.")
    with st.expander("Generated pandas code (validated & sandboxed)"):
        st.code(ans.code or "# none", language="python")
        if ans.log:
            st.write(ans.log)
