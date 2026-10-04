# NL Spreadsheet Analyst

Upload a messy CSV, ask a question in English, get a 1-line insight + table + chart.

Pipeline: load/auto-clean -> LLM (Groq/OpenAI) writes pandas -> AST safety check + sandbox -> table + Plotly chart + insight.

## Run
    pip install -r requirements.txt
    python -m streamlit run app.py
Paste a Groq API key (free at console.groq.com, starts with gsk_) in the sidebar, upload a CSV, ask a question.

## CLI
    python analyst.py data.csv "Top 10 complaint types"
    python analyst.py data.csv --eval

## Test data
    python make_test_data.py   # creates messy_311.csv and prints correct answers

## Safety
Generated code is rejected if it has imports, dunder access, eval/exec/open, file writers, while loops
or function definitions. It runs with whitelisted builtins, whitelisted pd/np, on a copy of the data, with a timeout.
