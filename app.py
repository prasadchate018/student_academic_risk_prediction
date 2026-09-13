import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Absolute path targeting the pickle file in the root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "logistic.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Input mappings for categorical fields
CATEGORICAL_MAPPINGS = {
    "parental_education": {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3},
    "family_income": {"Low": 0, "Medium": 1, "High": 2},
    "extracurricular": {"No": 0, "Yes": 1},
    "internet_access": {"No": 0, "Yes": 1}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Classifier</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --accent: #38bdf8;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            background-color: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 600px;
            border: 1px solid var(--border);
        }
        h2 { text-align: center; color: var(--accent); margin-bottom: 20px; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px; }
        input, select {
            padding: 10px;
            border-radius: 6px;
            border: 1px solid var(--border);
            background-color: #0f172a;
            color: var(--text);
            font-size: 0.95rem;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
        button {
            grid-column: span 2;
            margin-top: 15px;
            padding: 12px;
            background-color: var(--accent);
            color: #0f172a;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            font-size: 1rem;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
            background-color: #0f172a;
            border: 1px solid var(--accent);
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Prediction</h2>
        <form action="/" method="POST" class="grid">
            <div class="form-group">
                <label>Attendance (%)</label>
                <input type="number" step="any" name="attendance" required>
            </div>
            <div class="form-group">
                <label>Study Hours</label>
                <input type="number" step="any" name="study_hours" required>
            </div>
            <div class="form-group">
                <label>Past Failures</label>
                <input type="number" name="past_failures" required>
            </div>
            <div class="form-group">
                <label>Assignments Completed (%)</label>
                <input type="number" step="any" name="assignments_completed_pct" required>
            </div>
            <div class="form-group">
                <label>Parental Education</label>
                <select name="parental_education">
                    {% for key in mappings['parental_education'] %}
                    <option value="{{ key }}">{{ key }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>Family Income</label>
                <select name="family_income">
                    {% for key in mappings['family_income'] %}
                    <option value="{{ key }}">{{ key }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>Extracurricular</label>
                <select name="extracurricular">
                    {% for key in mappings['extracurricular'] %}
                    <option value="{{ key }}">{{ key }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>Internet Access</label>
                <select name="internet_access">
                    {% for key in mappings['internet_access'] %}
                    <option value="{{ key }}">{{ key }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label>Previous Grade</label>
                <input type="number" step="any" name="previous_grade" required>
            </div>
            <div class="form-group">
                <label>Final Score</label>
                <input type="number" step="any" name="final_score" required>
            </div>
            <button type="submit">Predict Status</button>
        </form>

        {% if prediction %}
        <div class="result">
            Prediction: <span style="color: var(--accent);">{{ prediction }}</span>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        if model is None:
            return render_template_string(
                HTML_TEMPLATE, 
                mappings=CATEGORICAL_MAPPINGS, 
                prediction="Error: Model file 'logistic.pkl' not found."
            )

        try:
            raw_features = [
                float(request.form["attendance"]),
                float(request.form["study_hours"]),
                float(request.form["past_failures"]),
                float(request.form["assignments_completed_pct"]),
                CATEGORICAL_MAPPINGS["parental_education"][request.form["parental_education"]],
                CATEGORICAL_MAPPINGS["family_income"][request.form["family_income"]],
                CATEGORICAL_MAPPINGS["extracurricular"][request.form["extracurricular"]],
                CATEGORICAL_MAPPINGS["internet_access"][request.form["internet_access"]],
                float(request.form["previous_grade"]),
                float(request.form["final_score"])
            ]
            
            features_array = np.array([raw_features])
            pred_class = model.predict(features_array)[0]
            prediction = str(pred_class)
        except Exception as e:
            prediction = f"Error processing input: {str(e)}"

    return render_template_string(HTML_TEMPLATE, mappings=CATEGORICAL_MAPPINGS, prediction=prediction)

# Required entry point for Vercel WSGI resolution
if __name__ == "__main__":
    app.run()
