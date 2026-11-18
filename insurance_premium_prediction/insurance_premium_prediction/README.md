# Insurance Premium Prediction System

Predicts a customer's yearly insurance premium (`charges`) from age, sex, BMI,
number of children, smoking status and region using machine learning.

## Setup
```
pip install -r requirements.txt
python train_model.py      # trains models, saves model.pkl
streamlit run app.py       # launches the web app
```

To use real data, place Kaggle's `insurance.csv` in this folder before training.

## Files
- `train_model.py` - data loading, preprocessing, model comparison, saving best model
- `app.py` - Streamlit UI for predictions
- `requirements.txt` - dependencies
