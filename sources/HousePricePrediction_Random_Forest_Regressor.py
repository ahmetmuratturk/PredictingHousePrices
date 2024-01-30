import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a CSV file
file_path = 'HousePricePrediction.csv'   # Replace with the actual path to your CSV file
housing_df = pd.read_csv(file_path)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can customize the strategy based on your needs

# Impute missing values in features (X)
features = ['MSSubClass', 'LotArea', 'OverallCond', 'YearBuilt', 'BsmtFinSF2', 'TotalBsmtSF']
X = housing_df[features]
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Impute missing values in the target variable (y)
y = housing_df['SalePrice']
y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)

# Advanced Model using Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Create a pipeline for preprocessing
numeric_features = X_imputed.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', model)])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Model Evaluation
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Feature Importance
feature_importance = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
feature_names = np.array(features)

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
