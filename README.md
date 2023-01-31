# Bank Customer Churn
This is a team collaboration project relevant to Machine Learning and predictions based on various learning models.

## Table of Contents

- [Overview of Project](#overview-of-project)
  - [Team Member](#team-member)
  - [Topic Selection](#topic-selection)
  - [Purpose of Project](#purpose-of-project)
  - [Current Status](#current-status)
  - [Resources](#resources)
- [References](#references)

## Overview of Project

### Team Member

Andia, Chris, Joey, Liwen, and Parto (alphabetical order).

### Topic Selection

We began exploring different datasets to address the question of when a company should execute layoffs. Unfortunately, for IPOs, the datasets we found contained less than 500 rows of relevant data. This seemed too small to create a robust machine learning model. We continued to explore different datasets regarding layoffs, but also began exploring the idea of creating a project regarding customer churn. One question to answer regarding customer churn is, what are the factors that lead to a customer either continuing or terminating their involvement (subscription, account, etc.) with the company. After viewing some datasets on telecom and bank customer churn, it seemed that these datasets had sufficient dimmensions (such as tenure and credit score for the bank datasets) and many rows 1000< to create a model. We will continue exploring this idea during the second class of our Final Project. Here is the link of the original dataset selected for our deep dive, [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download).</br>

### Purpose and Context

- Examine historical dataset
- Identify potential causes for bank customer churn
- Develop Machine learning model to predict churn

### Current Status

- Topic selection has been finalized.

- Assessment of dataset and database management system (DBMS) is completed.

- Preprocessing dataset and EDA is 90% completed.

- Model Testing and Determination

### Resources

- Source code: [EDA.ipynb](./EDA.ipynb), [DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb).
- Source data: [Churn_Modelling_2.csv](./Resources/Churn_Modelling_2.csv) (source: [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download)).
- Database data: [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv).
- Image file: png files.
- Software: [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide), [Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html), [Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/), etc.

### Next Step

Added Andia's SVM and NN models.

## References

[Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide)  
[Scikit-learn User Guide: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)  
[Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/)  
[Ensemble methods](https://imbalanced-learn.org/stable/references/ensemble.html#)  


