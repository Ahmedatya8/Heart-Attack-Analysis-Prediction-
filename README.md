# Heart-Attack-Analysis-Prediction
## Overview:
This project focuses on analyzing and predicting the likelihood of a heart attack based on clinical features. Using a dataset of patient information, the project applies machine learning models to predict whether a patient is at risk of experiencing a heart attack. The features include various health indicators such as cholesterol levels, blood pressure, and age.

## Project Structure:
- Exploratory Data Analysis (EDA): Insights into the dataset using visualization and statistical summaries.
- Feature Engineering: Preprocessing and transforming the data to improve model performance.
- Hyperprameter Tuning.
- Modeling: Training various machine learning models to predict heart attack risks, including:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - Support Vector Machine (SVM)
    - Gradient Boosting Classifier
    - XGBoost Classifier
    - K-Nearest Neighbors Classifier (KNN)
- Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.
## Installation:
To install the required libraries for this project, run the following commands:
```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install datasist
pip install scikit-learn
pip install xgboost
```
## Usage:
1. Clone the repository:
``` 
git clone https://github.com/your-username/heart-attack-prediction.git
```
2. Navigate to the project directory:
```
cd heart-attack-prediction
```
3. Install the required libraries (see the installation section).
4. Run the notebook or Python scripts to explore and train models.
## Data:
The dataset contains clinical data for patients, with features such as:
- Age, sex, and other demographics.
- Resting blood pressure and cholesterol levels.
- Fasting blood sugar and maximum heart rate achieved.
- Presence of chest pain, ST depression, and slope of peak exercise.
## Models:
The following machine learning models were used:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- XGBoost Classifier
## Evaluation:
The models are evaluated using several metrics, including:
- Accuracy: The percentage of correct predictions.
- Precision: The proportion of true positives among all positive predictions.
- Recall: The ability of the model to detect all actual positives.
- F1 Score: A balance between precision and recall.
## Results:
The best-performing model in this project was Random Forest Classifier, which achieved an accuracy of 0.8852 . Further improvements can be made by optimizing hyperparameters or exploring additional feature engineering techniques.

## Contributing:
Contributions are welcome! Feel free to submit a pull request or raise an issue if you have suggestions.

## License:
This project is licensed under the MIT License.
