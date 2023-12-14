This is our final project for Data Preparation course at NEU

We are Group 2 and our team members:

**RESPONSIBILITIES**
- Thieu Ngoc Mai:
    - EDA previous_application and POS_Cash_balance
    - Feature engineering, preprocessing and modeling
    - Refactoring other members code and creating final repo

- Tran Phuong Anh:
    - EDA application train|test
    - Support feature engineering application train
    - Slide

- Truong Minh Hung:
    - EDA installment_payments and credit_card_balance
    - Finding documents supporting for feature engineering installment_payments and credit_card_balance
    - Slide

- Phan Anh Khoi:
    - EDA bureau and bureau_balance
    - Support feature engineering bureau
    - Slide

**PROJECT FOLDER STRUCTURE**
- `EDA`: Folder includes exploratory analysis of all tables
- `preprocessing`: Folder includes all feature engineering and data cleanning for all the tables
- `modeling`: Folders includes code of feature selection and and modeling.

**HOW TO USE THIS REPO**
1. Clone this repo to your local, change all the path in the repo to your path.
2. Run `pip install -r requirements.txt` in your terminal to install all the neccessary packages.
4. Read the EDA folder to understand the data
5. Run script `main.py` in the `preprocessing` folder and save the final data.
6. Run script `fit_model.py` in the `modeling` folder to create the final predictions.

**REFERENCES**
- https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821
- https://www.kaggle.com/c/home-credit-default-risk/discussion/58332
- https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction
- https://www.kaggle.com/code/hikmetsezen/blend-boosting-for-home-credit-default-risk
- https://github.com/yakupkaplan/Home-Credit-Default-Risk
- https://www.kaggle.com/code/codename007/home-credit-complete-eda-feature-importance
- https://www.kaggle.com/code/shaz13/magic-of-weighted-average-rank-0-80/script
- https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
