# Spam Message Detector (Streamlit)

Classifies a message as SPAM or NOT SPAM using TF-IDF + Naive Bayes.
Trained on a small demonstration dataset (`data/spam_demo_dataset.csv`) - a learning demo, not a production spam filter.

## Run in VS Code
1. Unzip the folder and open it in VS Code (File > Open Folder).
2. Open a terminal (Terminal > New Terminal).
3. (Optional) Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Start the app: `python -m streamlit run app.py`
6. Your browser opens at http://localhost:8501 (press Ctrl+C in the terminal to stop).

## Using a bigger dataset
Replace `data/spam_demo_dataset.csv` with a CSV that has columns `label` (spam/ham) and `text`.
