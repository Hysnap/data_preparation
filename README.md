# Fake News or Real News

**Fake News or Real News** is a **data-driven dashboard** that helps users distinguish between real and fake news articles. This project implements **machine learning models, natural language processing (NLP), and data visualization** to provide insights into misinformation patterns and biases.

## 🌍 Live Dashboard

The dashboard is accessible at:  
**[Fake News Analysis Dashboard](https://YOUR_APP_NAME.herokuapp.com/)**  

---

## 📌 Project Objectives

This project was developed to:

- Provide a **public-facing dashboard** that visualizes **fake vs. real news trends**.
- Enable users to **analyze news articles** using **sentiment, subjectivity, and linguistic patterns**.
- Offer **a "News Scorer" tool** that predicts the likelihood of an article being fake.
- Display **geographical trends of misinformation** based on source locations.
- Enhance public awareness of **fake news detection techniques**.

---

## 📊 Dataset Content

- **Fake vs. Real News Dataset** ([Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection))
- **Misinformation & Fake News Text Dataset (79K)** ([Kaggle](https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k))
- **Geolocation Data** ([SimpleMaps](https://simplemaps.com/data/world-cities))

Additional details on data preparation are available in **[Notes on Data Preparation](notesondataprep.py)**.

---

## 🔍 Dashboard Features

The **Streamlit dashboard** provides multiple **interactive pages**:

### **1️⃣ Introduction**

- Overview of the **project goals** and **datasets used**.
- Course affiliation with **Code Institute** and **West Midlands Combined Authority**.
- Link to the **GitHub repository** ([introduction.py](introduction.py)).

### **2️⃣ Data Exploration**

- **Article Polarity vs. Title Polarity** (Scatter plot).
- **Article Count by Subject** (Bar chart).
- **Article Count by Source** (Comparison plot).
- **Media Type Distribution** (Bar chart).
- **Text vs. Title Length Distribution** ([visualisations.py](visualisations.py)).

### **3️⃣ Data Preprocessing**

- **Text cleaning, missing values, duplicates detection**.
- **Correlation heatmaps & outlier detection**.
- **Polarity and Subjectivity** analysis of titles vs. articles.
- **Variance and contradiction detection** ([datacleanliness.py](datacleanliness.py)).

### **4️⃣ Geolocation Analysis**

- **Heatmap of Fake vs. Real News Spread**.
- **Source distribution by continent, country, and region** ([Map_data_processing.py](Map_data_processing.py)).

### **5️⃣ Machine Learning Predictions**

- **RandomForestClassifier model** classifies articles as real or fake.
- **Regression-based "Realness Score" prediction**.
- **Model feature importance visualization** ([ML_model2.py](ML_model2.py)).

### **6️⃣ Fake News Trends & Alerts**

- **Time-series analysis** of fake news occurrences.
- **Top fake news sources and subjects**.
- **User-submitted article scoring tool**.

---

## 💑 Data Processing & Feature Engineering

### 🛠 **ETL & Transformation**

- **ETL Pipeline** ([ETL.py](ETL.py)) for:
  - Data extraction and combination.
  - Text preprocessing & missing values handling.
  - Sentiment analysis and subjectivity calculations.
  - Feature extraction (word count, media type, polarity).
  
- **Data Transformation** ([TRANSFORM.py](TRANSFORM.py)) for:
  - **Fake vs. Real news labeling**.
  - **Text sanitization and cleaning**.
  - **Date parsing and analysis**.
  - **Geolocation extraction**.

### 🔍 **Exploratory Data Analysis (EDA)**

- Outlier detection and bias identification.
- Variance analysis in title and article sentiment.
- Histogram distributions of fake vs. real news ([datacleanliness.py](datacleanliness.py)).

---

## 🤖 Machine Learning Model Development

**Models Developed:**

1. **Baseline Model**: RandomForestClassifier ([machinelearningmodel1_original.py](machinelearningmodel1_original.py)).
2. **Refactored Model**: Optimized feature selection & hyperparameters ([ML_Model_Chatgpt_refactor.py](ML_Model_Chatgpt_refactor.py)).
3. **Regression Model**: Assigns a "realness score" from 1 (fake) to 5 (real) ([ML_model2.py](ML_model2.py)).

**Key Techniques Used:**

- **Natural Language Processing (NLP)**:
  - Sentiment Analysis.
  - Subjectivity & Polarity Measurement.
  - Word Count & Readability Scores.
  
- **Feature Selection**:
  - Recursive Feature Elimination (RFE).
  - Variance Threshold.
  - Correlation Analysis.

- **Evaluation Metrics**:
  - Confusion Matrix.
  - Classification Report.
  - Feature Importance.

---

## 🌐 Dashboard Deployment

### **1️⃣ Local Setup**

```sh
pip install -r requirements.txt
streamlit run Real_Or_Dubious_News.py
```

### **2️⃣ Cloud Deployment (Streamlit)**

1. **Link GitHub repository**.
2. **Set environment variables (API keys, dataset paths, etc.)**.
3. **Deploy to Streamlit Cloud**.

---

## ⚠️ Challenges & Lessons Learned

### 🚨 **Challenges**

- **Dataset Bias**: The initial dataset contained strong biases (e.g., **source names were always identified for real news**).
- **Power BI Constraints**: Initial visualization approach abandoned due to license expiration.
- **Large File Sizes**: Managed using **compressed CSV storage**.

### ✅ **Solutions Implemented**

- **Bias Reduction**: Integrated **multiple datasets** to balance representation.
- **Streamlit Implementation**: Allowed **dynamic filtering & interactivity**.
- **Data Compression**: Used **ZIP archives** to store large files.

---

## 📌 Future Enhancements

- **Live News API Integration** (e.g., Snopes, PolitiFact).
- **Deepfake Image Detection**.
- **Community Flagging System** for fake news.
- **AI-Powered Fact-Checking Chatbot**.

---

## 📚 Credits

### 💜 **Datasets**

- [Kaggle: Fake News Detection](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection).
- [Misinformation & Fake News Dataset](https://www.kaggle.com/datasets/stevenpeutz/misinformation-fake-news-text-dataset-79k).

---

## 🙌 Acknowledgements

Special thanks to:

- **Code Institute & WMCA** for supporting this project.
- **Kaggle contributors** for dataset access.
- **Community feedback** on improving fake news detection.
- **ChatGPT, Gemini and CoPilot** for assistance with ideation, coding and refactoring.
