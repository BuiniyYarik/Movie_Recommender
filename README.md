# Movie Recommender

### Author:
##### Yaroslav Sokolov
##### Group: BS21-AI

### [GitHub Repository Link](https://github.com/BuiniyYarik/Movie_Recommender/tree/main)

### GitHub Repository Structure:
The repository has the following structure:
```
Movie Recommender
├── README.md         # The top-level README
│
├── data 
│   └── raw           # The original data used for models training and usage
│
├── models                              
│   ├── lr_predict.py       # The Logistic Regression model used to make predictions
│   └── svd_predict.py      # The SVD model used to make predictions
│
├── notebooks                               #  Jupyter notebooks
│   ├── 1.0-initial-data-exporation.ipynb                       # The initial data exploration
│   ├── 2.0-SVD-test-and-evaluation.ipynb                       # SVD model training and evaluation
│   ├── 3.0-LinearRegression-training-and-evaluation.ipynb      # Logistic Regression model training and evaluation
│   └── 4.0-recommenders-usage.ipynb                            # Recommenders usage example          
│
├── reports                        # Generated analysis in PDF format
│   └── Final_Report.pdf           # The final solution report
│
├── requirements.txt    # The requirements file for reproducing the analysis environment
│     
├── utils                   # Utility scripts
│   └── svd_utils.py        # Utility functions for SVD model
│                 
└── benchmark                              # Scripts to evaluate the models using hit ratio metric
    ├── calculate_hit_ratio_lr.py          # Evaluation of the Logistic Regression model
    └── calculate_hit_ratio_svd.py         # Evaluation of the SVD model
```

### How to use the repository:
1. Clone the repository:
```
git clone
```
2. Install the requirements:
```
pip install -r requirements.txt
```
3. Run the "svd_predict.py" script to make predictions using SVD model:
```
python models/svd_predict.py
```
4. Run the "lr_predict.py" script to make predictions using Logistic Regression model:
```
python models/lr_predict.py
```
