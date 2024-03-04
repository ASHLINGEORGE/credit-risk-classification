# Credit Risk Classification Analysis

## Overview of the Analysis
The objective of this analysis is to build a predictive model capable of accurately categorizing loan applicants into two classes: 'healthy loan' and 'high-risk loan'. This categorization is based on a variety of application factors, including loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, pre-defined derogatory marks, with the loan status as the target variable.

The target variable, loan status, was binary, comprising classes for healthy (0) and high-risk (1) loans. The dataset exhibited class imbalance, with a higher prevalence of healthy loans compared to high-risk loans, posing a challenge for accurate model training. To mitigate this issue, we employed logistic regression as our primary classification algorithm and implemented the RandomOverSampler technique. This resampling method artificially balanced the dataset by oversampling the minority class (high-risk loans), thereby improving the model's ability to accurately predict both classes and mitigate the effects of class imbalance.


## Results

**Machine Learning Model 1: Logistic Regression**

- Balanced Accuracy: 96.798%
- Precision for healthy loans (class 0): 1.00
- Recall for healthy loans (class 0): 0.99
- F1-score for healthy loans (class 0): 1.00
- Precision for high-risk loans (class 1): 0.84
- Recall for high-risk loans (class 1): 0.94
- F1-score for high-risk loans (class 1): 0.89

**Machine Learning Model 2: RandomOverSampler**

- Balanced Accuracy: 99.35%
- Precision for healthy loans (class 0): 1.00
- Recall for healthy loans (class 0): 0.99
- F1-score for healthy loans (class 0): 1.00
- Precision for high-risk loans (class 1): 0.84
- Recall for high-risk loans (class 1): 0.99
- F1-score for high-risk loans (class 1): 0.91

## Summary

Based on the results of the machine learning models:

* The logistic regression model with RandomOverSampler achieved a slightly higher balanced accuracy of 99.35% compared to the logistic regression model alone, which had a balanced accuracy of 96.798%.
The RandomOverSampler technique notably improved the recall for high-risk loans in the logistic regression model, resulting in a higher F1-score for high-risk loans at 0.91 compared to 0.89 in the logistic regression model alone.
* The choice of model depends on the specific problem we are trying to solve. If accurately identifying both healthy and high-risk loans is paramount, the logistic regression model with RandomOverSampler is recommended due to its superior overall performance, particularly in identifying high-risk loans.
However, if interpretability is more crucial and there is a tolerance for slightly lower performance in identifying high-risk loans, the logistic regression model alone could still be considered a viable option.
In summary, the logistic regression model with RandomOverSampler is recommended for its enhanced performance in accurately predicting both healthy and high-risk loans. However, the choice ultimately depends on the priorities and constraints of the problem at hand.