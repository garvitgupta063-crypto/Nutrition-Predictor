# Nutrition prediction app

This project uses the supplied scikit-learn artifacts:

- `model.pkl` — linear regression model
- `model1.pkl` — fitted `StandardScaler`

It expects these inputs in this exact order: `total_calories`, `fat_grams`,
`saturated_fat_grams`, `carb_grams`, and `sugar_grams`.

## Run it

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The original model was saved with scikit-learn 1.6.1. The requirements use
the compatible 1.8 release so installation works with Python 3.14.

The browser interface will open automatically. Enter non-negative nutrition
values and select **Predict**.

## Use from Python

```python
from predictor import NutritionPredictor

predictor = NutritionPredictor()
result = predictor.predict_one({
    "total_calories": 250,
    "fat_grams": 12,
    "saturated_fat_grams": 4,
    "carb_grams": 30,
    "sugar_grams": 10,
})
print(result)
```
