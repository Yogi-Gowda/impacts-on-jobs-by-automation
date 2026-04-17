# 🤖 AI Job Automation Risk Prediction System

## Project Overview
 
**Project Domain:** Predicting AI Impact on Jobs and Future Automation Risk  
**SDG Goal:** 17 - Partnerships for the Goals

## 📋 Abstract

Artificial Intelligence (AI) and automation are transforming the global job market. This project uses machine learning to predict which jobs are at risk of automation, helping workers and organizations prepare for the future of work.

## 🎯 Objectives

- Predict automation risk for different occupations
- Analyze job characteristics and their relationship with automation
- Classify jobs into Low, Medium, and High automation risk categories
- Provide actionable insights for career development

## 📊 Dataset

- **Source:** [Kaggle - AI Impact on Job Market (2024-2030)](https://www.kaggle.com/datasets/sahilislam007/ai-impact-on-job-market-20242030)
- **Size:** 30,000 rows × 13 columns
- **Features:**
  - Job Title, Industry, Location
  - Median Salary (USD)
  - Required Education
  - Experience Required (Years)
  - Job Openings (2024 & 2030)
  - Remote Work Ratio (%)
  - Automation Risk (%)
  - Gender Diversity (%)

## 🛠️ Methodology

### Machine Learning Approach

**Type:** Supervised Learning (Classification)

**Algorithms Implemented:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. K-Nearest Neighbors (KNN)
7. Support Vector Machine (SVM)
8. Naive Bayes

### Target Variable

Automation Risk is categorized into three classes:
- **Low Risk:** 0-33%
- **Medium Risk:** 34-66%
- **High Risk:** 67-100%

<!-- ## 🚀 Project Structure

```
AI_Job_Automation_Project/
│
├── scripts/
│   ├── 00_data_preprocessing.py     # Data Preprocessing ⭐ NEW!
│   ├── 01_eda_analysis.py           # Exploratory Data Analysis
│   ├── 02_feature_engineering.py    # Feature Engineering Pipeline
│   ├── 03_model_training.py         # Model Training & Evaluation
│   └── run_all.py                   # Master script to run all ⭐ NEW!
│
├── data/
│   ├── preprocessed_data.csv        # Clean data ⭐ NEW!
│   ├── preprocessing_report.txt     # Preprocessing report ⭐ NEW!
│   ├── X_train.csv                  # Training features
│   ├── X_test.csv                   # Test features
│   ├── y_train.csv                  # Training labels
│   ├── y_test.csv                   # Test labels
│   ├── processed_dataset.csv        # Complete processed data
│   └── model_comparison.csv         # Model performance metrics
│
├── models/
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   ├── label_encoders.pkl           # Categorical encoders
│   ├── feature_names.pkl            # Feature list
│   └── model_info.pkl               # Model metadata
│
├── visualizations/
│   ├── 00_outlier_detection.png     # Outlier analysis ⭐ NEW!
│   └── [Generated plots from EDA and modeling]
│
├── deployment/
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This file
│
└── app.py                           # Streamlit web application
```
-->
## 💻 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd AI_Job_Automation_Project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r deployment/requirements.txt
```

### Step 4: Run the Pipeline

```bash
# Method 1: Run all steps automatically
python scripts/run_all.py

# Method 2: Run each step individually
# 0. Data Preprocessing
python scripts/00_data_preprocessing.py

# 1. Run EDA
python scripts/01_eda_analysis.py

# 2. Run Feature Engineering
python scripts/02_feature_engineering.py

# 3. Train Models
python scripts/03_model_training.py
```

### Step 5: Launch Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`
<!--
## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click "Deploy"

### Option 2: Render

1. Create a `render.yaml` file:

```yaml
services:
  - type: web
    name: job-automation-predictor
    env: python
    buildCommand: pip install -r deployment/requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

2. Push to GitHub
3. Connect Render to your repository
4. Deploy

### Option 3: Heroku

1. Create `Procfile`:

```
web: sh setup.sh && streamlit run app.py
```

2. Create `setup.sh`:

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```
-->
3. Deploy using Heroku CLI

## 📊 Features

### 1. Data Exploration
- Interactive visualizations
- Statistical analysis
- Filter by industry, location, risk level
- Distribution analysis

### 2. Model Performance
- Compare 8 different ML algorithms
- Detailed metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrices
- Feature importance analysis

### 3. Prediction Tool
- Input job characteristics
- Get real-time automation risk predictions
- View confidence scores
- Receive personalized recommendations

### 4. Insights & Recommendations
- Industry-wise analysis
- Education impact on automation risk
- Salary trends
- Career development strategies

## 🎯 Results

### Best Model Performance

[To be filled after running the training pipeline]

- **Best Model:** [Model Name]
- **Accuracy:** [XX.XX%]
- **F1-Score:** [XX.XX%]
- **Precision:** [XX.XX%]
- **Recall:** [XX.XX%]

### Key Insights

1. **Industry Impact:** Manufacturing and Retail face higher automation risk
2. **Education Factor:** Higher education correlates with lower automation risk
3. **Salary Correlation:** Higher salaries typically indicate lower automation risk
4. **Remote Work:** Jobs with higher remote work ratios show varied risk levels

## 🔧 Technical Stack

- **Programming Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Cloud / Render / Heroku

## 📝 Usage Examples

### Making a Prediction

```python
# Load the model
import joblib
model = joblib.load('models/best_model.pkl')

# Prepare features (example)
features = [...]  # Your feature vector

# Predict
prediction = model.predict([features])
probability = model.predict_proba([features])

print(f"Risk Category: {prediction[0]}")
print(f"Confidence: {probability[0][prediction[0]] * 100:.2f}%")
```

## 📈 Future Enhancements

- [ ] Add more advanced models (Neural Networks, LightGBM)
- [ ] Implement SHAP values for better interpretability
- [ ] Add real-time data updates
- [ ] Include job recommendation system
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] API for integration with other systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Yogesh K**  
SRN: PES1PG25CA440  
PES University  
Email: [Your Email]  
LinkedIn: [Your LinkedIn]  
GitHub: [Your GitHub]

## 🙏 Acknowledgments

- PES University for project guidance
- Kaggle for the dataset
- Open-source community for tools and libraries

## 📚 References

1. World Economic Forum - "The Future of Jobs Report 2023"
2. McKinsey Global Institute - "Jobs Lost, Jobs Gained"
3. MIT Technology Review - "AI and the Future of Work"
4. Scikit-learn Documentation
5. Streamlit Documentation

## 📞 Support

For questions or support, please:
- Open an issue on GitHub
- Email: yogeshkumar20369@gmail.com
- LinkedIn: 

---

**Note:** This project is part of academic coursework at PES University and aligns with SDG Goal 17: Partnerships for the Goals.

**Last Updated:** April 2026
