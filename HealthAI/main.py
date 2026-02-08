import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DiseaseRiskPredictor:
    def __init__(self, data_path='health_risk_dataset.csv'):
        self.models = {}
        self.scalers = {}

        self.diseases = [
            'diabetes', 'hypertension',
            'heart_disease', 'obesity', 'depression'
        ]

        self.features = [
            'age', 'gender', 'bmi',
            'blood_pressure_sys', 'blood_pressure_dia',
            'cholesterol', 'glucose',
            'smoking_years', 'alcohol_consumption',
            'physical_activity', 'sleep_hours',
            'family_history_diabetes',
            'family_history_heart',
            'stress_level'
        ]

        self.targets = {
            'diabetes': 'has_diabetes',
            'hypertension': 'has_hypertension',
            'heart_disease': 'has_heart_disease',
            'obesity': 'has_obesity',
            'depression': 'has_depression'
        }

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, data_path)

        print("📊 Обучение моделей...")
        self.load_and_train(data_path)
        print("✅ Обучение завершено")

    def load_and_train(self, data_path):
        df = pd.read_csv(data_path)
        X = df[self.features]

        for disease, target in self.targets.items():
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=300,
                random_state=42,
                early_stopping=True
            )

            model.fit(X_train_scaled, y_train)

            self.models[disease] = model
            self.scalers[disease] = scaler

    def predict_from_dict(self, user_data):
        df = pd.DataFrame([user_data])
        results = {}

        for disease in self.diseases:
            X_scaled = self.scalers[disease].transform(df[self.features])
            prob = self.models[disease].predict_proba(X_scaled)[0, 1]

            if prob < 0.3:
                risk = "low"
            elif prob < 0.6:
                risk = "medium"
            elif prob < 0.8:
                risk = "high"
            else:
                risk = "very-high"

            results[disease] = round(prob * 100, 2), risk

        return results
