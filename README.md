![Screenshot_25](https://github.com/user-attachments/assets/3e2cd9b2-fa52-48c2-a3f4-b36371d594e2)

# Heart Disease Prediction Model

## Overview
This project aims to predict the likelihood of heart disease based on health and lifestyle factors such as **BMI**, **smoking habits**, **physical activity**, **diabetes status**, and other medical history. The prediction model is built using **Logistic Regression**, and the application is deployed using **Flask** for a simple web interface.

## Dataset
The dataset contains multiple health-related attributes, such as:
- **BMI**: Body Mass Index (a measure of body fat based on height and weight).
- **Smoking**: Whether the person is a smoker.
- **Alcohol Drinking**: Whether the person consumes alcohol.
- **Stroke**: Whether the person has had a stroke.
- **Physical Activity**: Whether the person has been physically active.
- **GenHealth**: Self-reported health status.
- **SleepTime**: Average sleep time per day.
- **Diabetes**: Whether the person has diabetes.
- **AgeCategory**, **Sex**, **Race**, **Kidney Disease**, **Asthma**, etc.

## Project Structure
```
├── app.py                   # Flask app for web interface
├── train_model.py            # Script to train and save the model
├── heart_disease_model.pkl   # Saved Logistic Regression model
├── templates/
│   ├── index.html            # Web form for input
│   └── result.html           # Displays prediction result
├── static/
│   └── style.css             # Optional styling for the web app
├── Heart Disease.csv         # Dataset used for training
├── requirements.txt          # Dependencies for the project
└── README.md                 # Project documentation (this file)
```

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Install Dependencies
Ensure you have Python installed. Install the required packages by running:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
To train the model, run the following command:
```bash
python train_model.py
```
This will preprocess the dataset, train the logistic regression model, and save it as `heart_disease_model.pkl`.

### 4. Run the Flask App
Start the Flask server by running:
```bash
python app.py
```
Open your web browser and navigate to `http://127.0.0.1:5000/` to access the app.

### 5. Predict Heart Disease
Use the web form to input health metrics like BMI, physical activity, and smoking habits, and the app will predict whether the person is at risk of heart disease.

## Features
- **Data Preprocessing**: Categorical encoding and normalization of features.
- **Logistic Regression**: A classification model to predict the likelihood of heart disease.
- **Web Interface**: Built using Flask, allowing users to input health details and receive predictions.

## Model Performance
The Logistic Regression model was evaluated using metrics like accuracy, precision, recall, and F1-score. The performance can be improved with hyperparameter tuning or trying different models like Random Forest or XGBoost.

## Dependencies
- Flask
- Pandas
- Scikit-learn
- Joblib
- TfidfVectorizer (for future use)

Install dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- **Model Improvements**: Experiment with other machine learning models like Random Forest or XGBoost.
- **Handling Imbalanced Data**: Implement methods like SMOTE or class weighting to deal with imbalanced datasets.
- **Deployment**: Deploy the model on cloud platforms such as Heroku or AWS.

## License
This project is licensed by Isbat Uddin

---
