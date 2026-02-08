from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from main import DiseaseRiskPredictor
import json
import os
import hashlib
import datetime

app = Flask(__name__)
app.secret_key = "super-secret-key"

predictor = DiseaseRiskPredictor()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_json(USERS_FILE)
        login = request.form["login"]
        password = request.form["password"]

        if login in users:
            return "Пользователь уже существует"

        users[login] = hash_password(password)
        save_json(USERS_FILE, users)

        session["user"] = login
        return redirect(url_for("profile"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_json(USERS_FILE)
        login = request.form["login"]
        password = hash_password(request.form["password"])

        if users.get(login) == password:
            session["user"] = login
            return redirect(url_for("profile"))

        return "Неверный логин или пароль"

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    history = load_json(HISTORY_FILE).get(session["user"], [])
    return render_template("profile.html", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    user_data = {
        'age': float(data['age']),
        'gender': int(data['gender']),
        'bmi': float(data['bmi']),
        'blood_pressure_sys': float(data['bp_sys']),
        'blood_pressure_dia': float(data['bp_dia']),
        'cholesterol': float(data['cholesterol']),
        'glucose': float(data['glucose']),
        'smoking_years': float(data['smoking']),
        'alcohol_consumption': float(data['alcohol']),
        'physical_activity': float(data['activity']),
        'sleep_hours': float(data['sleep']),
        'family_history_diabetes': int(data['family_diabetes']),
        'family_history_heart': int(data['family_heart']),
        'stress_level': float(data['stress'])
    }

    raw_results = predictor.predict_from_dict(user_data)

    translate = {
        'diabetes': 'Диабет',
        'hypertension': 'Гипертония',
        'heart_disease': 'Болезни сердца',
        'obesity': 'Ожирение',
        'depression': 'Депрессия'
    }

    response = []
    for key, (prob, risk) in raw_results.items():
        response.append({
            "name": translate[key],
            "probability": prob,
            "risk": risk
        })

    if "user" in session:
        history = load_json(HISTORY_FILE)
        history.setdefault(session["user"], []).append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "result": response
        })
        save_json(HISTORY_FILE, history)

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
