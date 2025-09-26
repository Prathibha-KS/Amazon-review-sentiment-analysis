Summary:

Amazon Review Sentiment Analysis is a machine learning project that classifies Amazon product reviews as positive or negative using NLP techniques. It includes data preprocessing, a Logistic Regression model, and an interactive Streamlit web app with visualizations like word clouds and sentiment distributions.

#Features:

Sentiment classification (Positive/Negative)
Data preprocessing with NLP
Logistic Regression model
Streamlit web app with live predictions
Visualizations: word clouds, sentiment distribution

#Install dependencies
pip install -r requirements.txt

#Run preprocessing (creates clean CSVs)
python load_and_preprocess.py

#Train & evaluate the model
python logistic_regression_model.py

#Launch the Streamlit app
python -m streamlit run app.py


## Installation & Setup

```bash
# Clone repo
git clone https://github.com/Prathibha-KS/Amazon-review-sentiment-analysis.git
cd Amazon-review-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
python -m streamlit run app.py
