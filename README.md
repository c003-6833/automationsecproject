# automationsecproject
# Machine Learning-Based Phishing URL Detection
**A URL Structure and Domain-Age Approach**  
Amy O'Neill - C00306833 - CW_KCCYB_B - South East Technological University

Classifies a URL as phishing or legitimate using 13 lexical features as well as domain age, without any blacklist. Compares Random Forest, Logistic Regression and a simple blacklist baseline.

## Run
step 1: pip install -r requirements.txt
step 2: python phishing_detector.py 
step 3: python phishing_detector.py --predict "http://secure-login.example.xyz/a@b" --age 12 #to view one URL, --age says the domain age is 12 days old

note: `--age` is the domain age in days (from WHOIS); use -1 if unknown.

## Data
`dataset_phishing.csv` is the benchmark dataset from Hannousse & Yahiouche (2021),
*Engineering Applications of Artificial Intelligence*, 104, 104347 (11,430 URLs, balanced).
Only the `url`, `domain_age` and `status` columns are used.

## Output
Running the program creates a `results/` folder containing:
- Charts: `confusion_matrix.png`, `feature_importance.png`, `model_comparison.png`
- Result tables: `all_results.csv`, `feature_importance.csv`, `misclassified_urls.csv`
- The saved model: `rf_model.joblib`

The text output from the run is saved in `results_console.txt`.