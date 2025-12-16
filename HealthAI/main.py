import pandas as pd
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

        print("📊 Обучение моделей...")
        self.load_and_train(data_path)
        print("✅ Обучение завершено\n")

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

            acc = model.score(scaler.transform(X_test), y_test)
            print(f"{disease:<15} | точность: {acc:.2%}")

            self.models[disease] = model
            self.scalers[disease] = scaler

    def predict_custom_patient(self):
        print("\n" + "=" * 60)
        print("ВВОД ДАННЫХ ПАЦИЕНТА")
        print("=" * 60)

        user_data = {
            'age': float(input("Возраст: ")),
            'gender': int(input("Пол (0-муж, 1-жен): ")),
            'bmi': float(input("ИМТ: ")),
            'blood_pressure_sys': float(input("Систолическое давление: ")),
            'blood_pressure_dia': float(input("Диастолическое давление: ")),
            'cholesterol': float(input("Холестерин: ")),
            'glucose': float(input("Глюкоза: ")),
            'smoking_years': float(input("Стаж курения (лет): ")),
            'alcohol_consumption': float(input("Алкоголь (ед./нед): ")),
            'physical_activity': float(input("Физическая активность (ч/нед): ")),
            'sleep_hours': float(input("Сон (часы): ")),
            'family_history_diabetes': int(input("Диабет у родственников (0/1): ")),
            'family_history_heart': int(input("Болезни сердца у родственников (0/1): ")),
            'stress_level': float(input("Уровень стресса (0-10): "))
        }

        self.display_predictions(self._predict(user_data))

    def _predict(self, user_data):
        df = pd.DataFrame([user_data])
        results = {}

        for disease in self.diseases:
            X_scaled = self.scalers[disease].transform(df[self.features])
            prob = self.models[disease].predict_proba(X_scaled)[0, 1]

            if prob < 0.3:
                risk, color = "НИЗКИЙ", "🟢"
            elif prob < 0.6:
                risk, color = "СРЕДНИЙ", "🟡"
            elif prob < 0.8:
                risk, color = "ВЫСОКИЙ", "🟠"
            else:
                risk, color = "ОЧЕНЬ ВЫСОКИЙ", "🔴"

            results[disease] = (prob, risk, color)

        return results

    def display_predictions(self, results):
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ АНАЛИЗА")
        print("=" * 60)

        translate = {
            'diabetes': 'Диабет',
            'hypertension': 'Гипертония',
            'heart_disease': 'Болезни сердца',
            'obesity': 'Ожирение',
            'depression': 'Депрессия'
        }

        for disease, (prob, risk, color) in sorted(
                results.items(), key=lambda x: x[1][0], reverse=True):
            print(f"{translate[disease]:<20} | {prob:6.2%} | {color} {risk}")


if __name__ == "__main__":
    predictor = DiseaseRiskPredictor()
    predictor.predict_custom_patient()
