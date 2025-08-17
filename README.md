End-to-end mini-project covering:

1. Web scraping of British Airways reviews, cleaning, sentiment analysis (VADER), word counts & bigrams, and exploratory plots. 

2. Predictive modeling on a structured customer_booking dataset: EDA, feature engineering, imbalance-aware classifiers (LogReg/XGBoost), threshold tuning, and a simple regression for flight_duration.

Task 1 — Reviews scraping & NLP 
Scraper: paginated reviews → BA_reviews.csv (review_text, seat_type, date_flown, recommend, country).

Cleaning: remove “✅Trip Verified | Not Verified” tags, strip symbols, create trip_verified_flag.

Sentiment: VADER (Positive/Negative/Neutral via compound score).

Text features: custom stop-words, tokenization, lemmatization; word counts & bigrams with CountVectorizer.

Visuals:

Seat-type distribution
<img width="784" height="484" alt="image" src="https://github.com/user-attachments/assets/1361aca8-bb05-4cbf-b2b9-babdf60d7e1b" />

Sentiment by seat type (stacked countplot)
<img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/70ef414c-351e-4dc0-bb8e-1bce7356e2e9" />

Recommendation pie (yes vs no)
<img width="432" height="425" alt="image" src="https://github.com/user-attachments/assets/a754f445-4b4c-46cf-9fa3-39a5654caaf5" />

WordClouds (airport codes from routes; positive/negative clouds)
<img width="794" height="429" alt="image" src="https://github.com/user-attachments/assets/0505751a-05d5-44d7-b044-c8a284895f7f" />
<img width="794" height="429" alt="image" src="https://github.com/user-attachments/assets/428389ca-c369-402d-a1fa-bb57c1ad16d1" />

Top bigrams bar chart
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/25ea5519-01a0-46dd-9c6c-368bdff8a2c6" />


Task 2 — Predictive modeling (customer_bookings) 
EDA: distributions, purchase lead vs completion, preferences vs completion, top routes/origins.

Preprocessing:

Map flight_day to integers (Mon=1 … Sun=7)

Target-mean encoding for high-card categories (e.g., route) fit on train only (no leakage)

One-hot for low-card categoricals; scale/impute numerics.

Models:

Logistic Regression (class_weight='balanced'), probability threshold tuning (PR curve / F1).

XGBoost with scale_pos_weight; optional RandomizedSearchCV targeting PR-AUC.

Regression (predict flight_duration) + per-route mean baseline and GroupKFold by route.

Explainability: top coefficients (LogReg) and (optional) SHAP / feature importance (XGB).

🧪 Reproducibility tips

Set pages and page_size at the top of the scraper cell to control how many reviews you pull. Respect robots.txt and add polite delays if you scale up.

For modeling, keep a fixed random_state and stratified splits for classification.

When reporting regression, include both the standard split and GroupKFold by route (unseen routes) to show generalization.

Save figures to reports/figures/ if you want them versioned.

📊 Example outputs

Booking completion distribution, Trip type distribution
<img width="684" height="383" alt="image" src="https://github.com/user-attachments/assets/65063186-8f6b-4cea-8b35-9aa27d1ded5a" />
<img width="684" height="384" alt="image" src="https://github.com/user-attachments/assets/a5bf2cfa-b051-4696-a5b5-08856f20af22" />
<img width="837" height="684" alt="image" src="https://github.com/user-attachments/assets/0bc1a01c-23ce-4afb-86fe-48df923607d3" />

Regression: Predicted vs Actual plot, MAE/RMSE in minutes, per-route baseline metrics.
<img width="584" height="584" alt="image" src="https://github.com/user-attachments/assets/d3cc28a9-bbda-443b-8be7-bea845838adb" />
<img width="684" height="384" alt="image" src="https://github.com/user-attachments/assets/1497dbb1-baf1-4815-9bf7-e9cb3c54da02" />



 
