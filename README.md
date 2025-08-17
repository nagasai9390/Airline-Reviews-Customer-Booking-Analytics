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

Sentiment by seat type (stacked countplot)

Recommendation pie (yes vs no)

WordClouds (airport codes from routes; positive/negative clouds)

Top bigrams bar chart

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
Seat types distribution, Sentiment by seat type, Recommendation pie, Top airport codes WordCloud, Top bigrams.

Classification reports (default & threshold-tuned), ROC & PR curves.

Regression: Predicted vs Actual plot, MAE/RMSE in minutes, per-route baseline metrics.

(Screenshots/PNGs can be placed under reports/figures/ and referenced here.)

⚠️ Notes on scraping
This project is for educational use. If you scrape at scale, add:

request headers & polite delays,

caching,

and follow the website’s terms of service.

Avoid overloading the source site.

 
