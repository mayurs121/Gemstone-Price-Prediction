
# Gemstone Price Prediction By Mayur Redekar 

## Azure Deployment Link: 
https://gemstonepriceprediction.azurewebsites.net/predict 

## Screenshot Of UI:
![gem_sc](https://github.com/mayurs121/Gemstone-Price-Prediction/assets/101388775/f2166c88-c5eb-4f68-8c07-3b6263c006ed)

## About The Data:
The goal is to predict price of given diamond (Regression Analysis).

There are 10 independent variables (including id):
* id : unique identifier of each diamond
* carat : Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
* cut : Quality of Diamond Cut
* color : Color of Diamond
* clarity : Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these     characteristics under 10-power magnification.
* depth : The depth of diamond is its height (in millimeters) measured from the culet (bottom tip) to the table (flat, top surface)
* table : A diamond's table is the facet which can be seen when the stone is viewed face up.
* x : Diamond X dimension
* y : Diamond Y dimension
* z : Diamond Z dimension

Target variable:

* price: Price of the given Diamond. 

## Approach for the project

1. Data Ingestion :

* The data is initially read as CSV during the Data Ingestion phase.
* The data is then divided into training and testing and saved as a csv file.

2. Data Transformation :

* A ColumnTransformer Pipeline is established during this stage.
* SimpleImputer is used using the median approach first for Numeric Variables, and then Standard Scaling is applied to the numeric data.
* Ordinal encoding is carried out after this data is scaled with Standard Scaler for Categorical Variables using SimpleImputer's most frequent technique.
* A pickle file is used to save this preprocessing.


3. Model Training :

* The base model is evaluated during this phase. 
* Catboost Regressor was the model that performed the best.
* The catboost and knn models are then tuned by hyperparameter tuning.
* The prediction of the catboost, xgboost combined into a single voting predictor.
* A pickle file is used to save this model.

4. Prediction Pipeline :

This pipeline transforms input data into a dataframe and includes a number of functions to load pickle files and forecast outcomes in Python.

5. Flask App development:

In a web application, a flask app is developed with a user interface for predicting gemstone prices.

## Exploratory Data Analysis Notebook 
Link: https://github.com/mayurs121/Gemstone-Price-Prediction/blob/main/notebook/EDA.ipynb 

## Model Training Approach 
Link: https://github.com/mayurs121/Gemstone-Price-Prediction/blob/main/notebook/model_training.ipynb






