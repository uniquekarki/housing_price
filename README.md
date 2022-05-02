Housing Price
==============================

Housing price ML model

### To run the project:
1. Install the requirement from requirements.txt.
2. Go to '/src'.
3. Run `main.py` file.

### Result:
1. Cleaned data is stored in `/data/processed` as `clean_test_data.csv` and `clean_train_data.csv`.
2. Predicted data is saved in `/data/prediction/prediction.csv`.
3. Models for categorical features encoding, scaling of numerical input features and output is stored in `/models/data-cleaning-models`
4. Final ML model is stored in `/models/final_gbr.pkl`

Project Organization
------------

    ├── LICENSE
    ├── Makefile                     <- Makefile with commands like `make data` or `make train`
    ├── README.md                    <- The top-level README for developers using this project.
    ├── data
    │   ├── prediction               <- Final result.
    │   ├── processed                <- The final, canonical data sets for modeling.
    │   └── raw                      <- The original, immutable data dump.
    │
    ├── docs                         <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                       <- Trained and serialized models, model predictions, or model summaries
    │   ├── data-cleaning-models     <- Models used for categorical feature encoding and feature scaling.
    │   └── ml-models                <- The final ML model used for prediction
    │
    ├── notebooks                    <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                   the creator's initials, and a short `-` delimited description, e.g.
    │                                   `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt             <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    │
    └── src                          <- Source code for use in this project.
        ├── dataCleaning.py          <- Final result.
        ├── main.py                  <- The final, canonical data sets for modeling.
        └── modelTraining.py         <- The original, immutable data dump.
        



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
