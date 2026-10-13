# 🏠 USA Housing Price Predictor

### AI-Powered Real Estate Price Estimation with Multiple Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

**USA Housing Price Predictor** is an end-to-end Machine Learning project that predicts house prices in the United States using the classic **USA Housing** dataset.  

The project trains and compares **11 different regression models** and provides a beautiful, interactive **Streamlit web application** for real-time price prediction, model comparison, and data exploration.

### ✨ Key Highlights

| Feature | Description |
|---------|-------------|
| 🧠 **11 ML Models** | Linear, Ridge, Lasso, ElasticNet, Polynomial, Random Forest, XGBoost, KNN, SVM, ANN, Huber |
| 🎯 **Best R² Score** | **0.9146** (Linear / Ridge / Lasso Regression) |
| 💻 **Interactive UI** | Modern dark-themed Streamlit dashboard with animations |
| 📊 **Model Comparison** | Side-by-side performance metrics + interactive charts |
| 📈 **Data Insights** | Price distribution, feature relationships, statistics |
| 🔒 **100% Local** | Runs completely offline — no data leaves your machine |

---

## 🚀 Live Demo

```bash
streamlit run app.py
```

> Open your browser at `http://localhost:8501`

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Web Framework:** Streamlit
- **ML Libraries:** Scikit-learn, XGBoost
- **Visualization:** Plotly Express
- **Data Handling:** Pandas, NumPy
- **Model Persistence:** Pickle

---

## 📁 Project Structure

```text
usa-housing-price-predictor/
│
├── app.py                          # Main Streamlit application
├── USA_Housing.csv                 # Dataset (5000 records)
├── model_results.csv               # Performance metrics of all models
│
├── LinearRegression_model.pkl
├── RidgeRegression_model.pkl
├── LassoRegression_model.pkl
├── ElasticNet_model.pkl
├── PolynomialRegression_model.pkl
├── RandomForest_model.pkl
├── XGBoost_model.pkl
├── KNN_model.pkl
├── SVM_model.pkl
├── ANN_model.pkl
├── RobustRegression_model.pkl
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

**Dataset:** USA Housing  
**Records:** 5,000  
**Target Variable:** `Price`

| Feature | Description |
|---------|-------------|
| `Avg. Area Income` | Average income of residents in the area |
| `Avg. Area House Age` | Average age of houses in the area |
| `Avg. Area Number of Rooms` | Average number of rooms |
| `Avg. Area Number of Bedrooms` | Average number of bedrooms |
| `Area Population` | Population of the area |

---

## 🏆 Model Performance

| Rank | Model | R² Score | MAE | Remarks |
|------|-------|----------|-----|---------|
| 🥇 | **Lasso Regression** | **0.9146** | $82,658 | Best overall |
| 🥈 | **Linear Regression** | **0.9146** | $82,658 | Extremely close |
| 🥉 | **Ridge Regression** | **0.9146** | $82,660 | Excellent |
| 4 | ElasticNet | 0.8780 | $99,127 | Good |
| 5 | Random Forest | 0.8774 | $98,918 | Strong ensemble |
| 6 | XGBoost | 0.8710 | $102,161 | Competitive |
| 7 | Polynomial (deg=4) | 0.8710 | $97,863 | Non-linear |
| 8 | KNN | 0.5114 | $198,086 | Moderate |
| 9 | ANN (MLP) | 0.5023 | $199,269 | Needs tuning |
| 10 | Huber (Robust) | 0.5011 | $199,466 | Robust to outliers |
| 11 | SVM | 0.0004 | $282,948 | Poor fit |

> **Insight:** On this dataset, simple linear models outperform complex tree-based and neural network models.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/usa-housing-price-predictor.git
cd usa-housing-price-predictor
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```text
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.15.0
xgboost>=1.7.0
statsmodels>=0.14.0
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 🖥️ How to Use the App

1. **Select a Model** from the sidebar (Linear / Ridge / Lasso recommended)
2. **Enter Feature Values**:
   - Average Area Income
   - Average House Age
   - Number of Rooms
   - Number of Bedrooms
   - Area Population
3. Click **🚀 Predict House Price**
4. View the predicted price with model performance metrics
5. Explore **Model Comparison** and **Data Insights** tabs

---

## 🎨 UI Features

- 🌑 Modern dark gradient theme
- ✨ Animated prediction result card
- 🃏 Glassmorphism cards with hover effects
- 📱 Fully responsive design
- 📊 Interactive Plotly charts
- 🏷️ Model icons & performance badges
- ⚡ Smooth animations and transitions

---

## 📈 Screenshots

> Add your screenshots here after running the app

| Prediction Tab | Model Comparison | Data Insights |
|----------------|------------------|---------------|
| ![Prediction](screenshots/prediction.png) | ![Comparison](screenshots/comparison.png) | ![Insights](screenshots/insights.png) |

---

## 🧠 Key Learnings

- Linear models can outperform complex ensembles on well-behaved tabular data
- Feature scaling and proper preprocessing are critical for models like SVM & SGD
- Streamlit makes it extremely easy to turn ML models into production-ready demos
- Model comparison is essential — the “best” model depends on the dataset

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV / Optuna
- [ ] Feature importance visualization
- [ ] SHAP explainability
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add more advanced models (CatBoost, LightGBM)
- [ ] User authentication & prediction history

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Gaurav**  
Machine Learning Project – Housing Price Prediction

---

<div align="center">

**⭐ If you found this project useful, please give it a star!**

`usa-housing-price-predictor`

</div>
