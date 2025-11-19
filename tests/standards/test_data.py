import pandas as pd

test_data = {
    "summaries_train": ["Cyber attacks increased", "Malware detected in network", "Phishing attempt blocked"],
    "summaries_test": ["Ransomware detected", "Unauthorized access attempt"],
    "nom_summaries_train": ["IT policy update", "New employee onboarding", "Security training"],
    "nom_summaries_test": ["Quarterly audit", "Software patch deployment"],
    "X_train": pd.DataFrame({"feature1": [1, 2, 1, 4]}),
    "y_train": pd.DataFrame({"target": [1, 0, 1, 1]}),
    "X_test": pd.DataFrame({"feature1": [4, 5]}),
    "y_test": pd.DataFrame({"target": [0, 1]}),
    "X_metr_train": pd.DataFrame({"metric1": [0.1, 0.2, 0.3, 0.2]}),
    "X_metr_test": pd.DataFrame({"metric1": [0.4, 0.5]}),
}
