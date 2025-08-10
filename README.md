# AI-ML-Internship-Task-3---Linear-Regression
 -Classification-with-Logistic-Regression
 Project: Binary Classification with Logistic Regression

This project builds a binary classifier to predict whether a breast cancer diagnosis is benign or malignant using the Breast Cancer Wisconsin dataset. The implementation uses Python with Scikit-learn, Pandas, and Matplotlib.

Objective

The main objective is to apply logistic regression for a binary classification task and evaluate its performance using various metrics. 
Process

1. Dataset: The Breast Cancer Wisconsin dataset from Scikit-learn's built-in datasets was used. [cite: 14]
2. Data Preprocessing: The dataset was split into training and testing sets. [cite_start]The features were then standardized to ensure the model performs accurately.
3.  Model Training: A logistic regression model was trained on the preprocessed data. [cite: 10]
4.  Evaluation: The model's performance was assessed using:
    Confusion Matrix 
    Precision and Recall
    ROC-AUC Score 
5. Threshold Tuning: The impact of adjusting the decision threshold on precision and recall was explored. 

Libraries Used

* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

How to Run**
1.  Clone the repository.
2.  Ensure you have Python and the required libraries installed.
3.  Run the `main.py` script. The script will train the model, print the evaluation metrics, and display the confusion matrix and ROC curve plots

 Interview Questions & Solutions
 
1. What assumptions does linear regression make?
Linear regression makes four key assumptions:

Linearity: The relationship between predictors and the target variable is linear.

Independence: The errors (residuals) are independent of each other.

Homoscedasticity: The errors have constant variance across all levels of the predictors.

Normality: The errors are normally distributed.

2. How do you interpret the coefficients?
A coefficient represents the average change in the target variable for a one-unit increase in its predictor variable, assuming all other predictors are held constant.

3. What is R² score and its significance?
The R-squared (R²) score is a statistical metric that indicates the proportion of the variance in the target variable that is explained by the model's predictors. Its significance lies in measuring the "goodness-of-fit" of the model. A score of 0.61 means that 61% of the price variation is accounted for by the model.

4. When would you prefer MSE over MAE?
You would prefer Mean Squared Error (MSE) over Mean Absolute Error (MAE) when you want to heavily penalize larger errors. Since MSE squares the error term, it gives a much larger weight to significant mistakes, which is useful when large errors are particularly undesirable.

5. How do you detect multicollinearity?
Multicollinearity (when predictors are highly correlated) can be detected using:

Correlation Matrix: A matrix showing correlation coefficients between all pairs of variables. High values (e.g., > 0.8) are a red flag.

Variance Inflation Factor (VIF): VIF measures how much a coefficient's variance is inflated due to correlation with other predictors. A VIF score over 5 or 10 is a common indicator of multicollinearity.

6. What is the difference between simple and multiple regression?
Simple Linear Regression uses only one independent variable to predict a target.

Multiple Linear Regression uses two or more independent variables to predict a target.

7. Can linear regression be used for classification?
No, it is not suitable for classification. Linear regression predicts continuous values (like price). Classification predicts discrete categories (like "Yes/No"). Algorithms like Logistic Regression are designed for this purpose.

8. What happens if you violate regression assumptions?
Violating the assumptions makes the model's results unreliable. The coefficients may be biased, and their standard errors will be inaccurate, leading to faulty conclusions about which predictors are truly significant.   
   

1.  Clone the repository.
2.  Ensure you have Python and the required libraries installed.
3.  Run the `main.py` script. The script will train the model, print the evaluation metrics, and display the confusion matrix and ROC curve plots.
