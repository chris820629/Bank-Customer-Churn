# Bank Customer Churn
This is a team collaboration project relevant to Machine Learning, Neural Networks, and predictions based on various learning models.

## Table of Contents

- [Overview of Project](#overview-of-project)
  - [Team Member](#team-member)
  - [Topic Selection](#topic-selection)
  - [Purpose of Project](#purpose-of-project)
  - [Current Status](#current-status)
- [Segment 1: Sketch It Out](#segment-1-sketch-it-out)
  - [Resources](#resources)
  - [Next Step](#next-step)
- [Segment 2: Build and Assemble](#segment-2-build-and-assemble)
- [References](#references)

## Overview of Project

This project is divided into three Segments: Segment 1, Segment 2, and Segment 3. A checkbox with checkmark in it indicates that the corresponding segment and tasks are completed. 

- ✅ Segment 1: Sketch It Out.
- ✅ Segment 2: Build and Assemble.
- 🟩 Segment 3: Put It All Together.

### Team Member

Andia, Chris, Joey, Liwen, and Parto (alphabetical order).

### Topic Selection

We began exploring different datasets to address the question of when a company should execute layoffs. Unfortunately, for IPOs, the datasets we found contained less than 500 rows of relevant data. This seemed too small to create a robust machine learning model. We continued to explore different datasets regarding layoffs, but also began exploring the idea of creating a project regarding customer churn. One question to answer regarding customer churn is, what are the factors that lead to a customer either continuing or terminating their involvement (subscription, account, etc.) with the company. After viewing some datasets on telecom and bank customer churn, it seemed that these datasets had sufficient dimensions (such as tenure and credit score for the bank datasets) and many rows over 10000 to create a learning model. We will continue exploring this idea during the second class of our Final Project. Here is the link to the original dataset selected for our deep dive, [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download).

Questions that our team plans to answer is how to predict churn rate of customers based on bank customers' involvement, the reason(s) why customers left, and whether Machine Learning or Neural Network models can help our stakeholders solve.

### Purpose of Project

This project was the final team project to cultivate collaboration, teamwork, and effective use of collaboration tools, such as GitHub, Database Management System (DBMS), and online story board/dashboard. During this project, we were also encouraged to focus on Machine Learning or Neural Network models, and apply those techniques to solve a real world case study. Here are a few systematic steps that we have performed to find the best performing solution(s).

- Examine historical dataset.
- Preprocess the dataset, including proper data cleaning, standardization, and scaling whichever necessary.
- Identify potential causes for bank customer churn.
- Develop Machine learning model to predict churn rate.

### Current Status

- ✅ Topic selection has been finalized: prediction of bank customer churn rate.

- ✅ Assessment of dataset and database management system (DBMS) is completed.
  - Our dataset consisted of 10000 rows and 14 columns. A few numeric columns contained some outliers as illustrated in Fig. 1(a)&ndash;(c).
  - A PostgreSQL database that stores two tables, called **main_df** and **clean_df**, was created and can be connected without problems from Python code of each Team Member. We documented some SQL queries for retrieving some data from the database ([DBMS_Analysis.ipynb](./DBMS_Analysis.ipynb)).

  <hr>
  <table><tr><td><img src='Data/CreditScore_boxplot.png' title='(a) Column CreditScore'></td><td><img src='Data/Age_boxplot.png' title='(b) Column Age'></td><td><img src='Data/NumOfProducts_boxplot.png' title='(c) Column NumOfProducts'></td></tr></table>

  **Fig. 1 Boxplots of several numerical columns containing some outliers: (a) Column CreditScore, (b) Column Age, and (c) Column NumOfProducts.**
  <hr>

- ✅ Preprocessing dataset and EDA is completed. The cleaned datasets, [Churn_Modelling_main.csv](./Data/Churn_Modelling_main.csv) and [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv), are stored in the GitHub repo and PostgreSQL database.

- 🟩 Model Testing and Determination.
  - Evaluation Machine Learning or Neural Network models that could effectively predict bank customer churn rate.
  - Optimization of our final models is ongoing.

## Segment 1: Sketch It Out

Our team discussed about our overall project, resources (datasets, technologies, software, ML/NN models, etc.), select a question/topic to focus on, and build a simple model. We then prototyped our team's idea by using either CSV or JSON files to connect the model to a fabricated database.

### Resources

The clean dataset can referred in [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv), in which we removed a few customers with unrealistically high credit scores using the boolean indexing method in Python.

- GitHub repository: [Bank-Customer-Churn](https://github.com/chris820629/Bank-Customer-Churn) for sharing our analysis details, datasets, and results.
- Source code: [EDA.ipynb](./EDA.ipynb), [BankCustomerChurn_Optimization.ipynb](./BankCustomerChurn_Optimization.ipynb), [Decision_Tree_Customer_Churn.ipynb](./Decision_Tree_Customer_Churn.ipynb).
- Source data: [Churn_Modelling_2.csv](./Resources/Churn_Modelling_2.csv) (source: [Churn of Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download)).
- Database data: [Churn_Modelling_main.csv](./Data/Churn_Modelling_clean.csv), [Churn_Modelling_clean.csv](./Data/Churn_Modelling_clean.csv).
- Fabricated DBMS: PostgreSQL ([PostgreSQL documentation](https://www.postgresql.org/docs/)).
- Image file: png files.
- Software: [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide), [Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html), [Python imbalanced-learn](https://pypi.org/project/imbalanced-learn/).
- Tableau dashboard: TBD.

### Next Step

We have been working on improving the accuracy and sensitivity of our learning models and preparing story/dashboard for presenting our data effectively.

- Tableau dashboard/story.
- Optimization of our learning models.
- Final touches.

## Segment 2: Build and Assemble

Our team performed some experiments to test and train our models, build the database that we will use for our final presentation, and create both our dashboard and presentation.

### Analysis Results

**Table 1. Condensed summary statistics of all learning models. Precision, Recall, and F1 score are the avg/total values (Used metrics: low &lt; 60%, good = 60&ndash;70%, very good = 70&ndash;90%, high &gt; 90%).**  
| ML method              | Dataset-Exited | Balanced accuracy score | Precision | Recall  | F1 score | Conclusion                                      |
| :--                    | :--:           |                     --: |       --: |     --: |      --: | :--                                             |
| BalancedRandomForest   | Original-0     |   0.784557              | 0.93      |    0.81 |  0.86    | Very good accuracy/recall/F1 score              |
|                        | Original-1     |   0.784557              | 0.51      |    0.76 |  0.61    | Very good accuracy/recall, good F1 score        |
| EasyEnsembleClassifier | Original-0     |   0.778234              | 0.93      |    0.80 |  0.86    | Very good accuracy/recall/F1 score              |
|                        | Original-1     |   0.778234              | 0.50      |    0.75 |  0.60    | Very good accuracy/recall, low F1 score         |
| AdaBoostClassifier     | Original-0     |   0.729186              | 0.88      |    0.96 |  0.92    | Very good accuracy, **high recall/F1 score**    |
|                        | Original-1     |   0.729186              | 0.78      |    0.49 |  0.61    | Very good accuracy, *low recall*, good F1 score |
| BalancedRandomForest   | NoFliers-0     |   0.755076              | 0.93      |    0.78 |  0.85    | Very good accuracy/recall/F1 score              |
|                        | NoFliers-1     |   0.755076              | 0.43      |    0.73 |  0.54    | Very good accuracy/recall, *low F1 score*       |
| EasyEnsembleClassifier | NoFliers-0     |   0.759114              | 0.93      |    0.77 |  0.84    | Very good accuracy/recall/F1 score              |
|                        | NoFliers-1     |   0.759114              | 0.43      |    0.75 |  0.54    | Very good accuracy/recall, *low F1 score*       |
| AdaBoostClassifier     | NoFliers-0     |   0.686034              | 0.88      |    0.96 |  0.92    | Very good accuracy, **high recall/F1 score**    |
|                        | NoFliers-1     |   0.686034              | 0.68      |    0.42 |  0.52    | Good accuracy, *low recall/F1 score*            |
| BalancedRandomForest   | Clean-0        |   0.772251              | 0.93      |    0.81 |  0.86    | Very good accuracy/recall/F1 score                 |
|                        | Clean-1        |   0.772251              | 0.48      |    0.74 |  0.58    | Very good accuracy/recall, *low F1 score*          |
| EasyEnsembleClassifier | Clean-0        |   0.771172              | 0.93      |    0.79 |  0.85    | Very good accuracy/recall/F1 score                 |
|                        | Clean-1        |   0.771172              | 0.47      |    0.75 |  0.58    | Very good accuracy/recall, *low F1 score*          |
| AdaBoostClassifier     | Clean-0        |   0.717164              | 0.88      |    0.95 |  0.92    | Very good accuracy, **high recall/F1 score**       |
|                        | Clean-1        |   0.717164              | 0.71      |    0.48 |  0.58    | Very good accuracy, *low recall/F1 score*          |

<hr>
<table><tr><td><img src='Data/BankCustomerChurn_main_df_FeatureImportance.png' title='(a) Sorted feature importance'></td><td><img src='Data/BankCustomerChurn_clean_df_FeatureImportance.png' title='(b) Sorted feature importance w/o outliers'></td></tr></table>

**Fig. 2 Sorted feature importances of (a) the original dataset and (b) the dataset after removal of unrealistically high credit scores.**
<hr>

## Segment 3: Put It All Together

We put the final touches on our models, database, and dashboard. Then create and deliver our final presentation to the class.

## References

[Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html#user-guide)  
[TensorFlow Documentation](https://www.tensorflow.org/guide/)  
[Scikit-learn User Guide - Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)  
[Scikit-learn User Guide - Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)  
[Matplotlib - Plot types](https://matplotlib.org/stable/plot_types/index.html)  
[Ensemble methods](https://imbalanced-learn.org/stable/references/ensemble.html#)  
[PostgreSQL documentation](https://www.postgresql.org/docs/)  
