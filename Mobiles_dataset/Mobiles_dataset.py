# https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('/home/sang/BaiTap/MachineLearning/Mobiles_dataset/Mobiles Dataset (2025).csv', encoding='latin1')


def remove_characters(value):
    if pd.isna(value) or value == '' or value == ' ':
        return np.nan
    str_number = re.findall(r'[\d,.]+', str(value))
    if len(str_number) == 0:
        return np.nan
    try:
        cleaned_numbers = [float(num.replace(',', '')) for num in str_number]
        return sum(cleaned_numbers)
    except ValueError:
        return np.nan


columns_to_clean = [
    'Mobile Weight', 'RAM', 'Front Camera', 'Back Camera',
    'Battery Capacity', 'Screen Size', 'Launched Price (Pakistan)',
    'Launched Price (India)', 'Launched Price (China)',
    'Launched Price (USA)', 'Launched Price (Dubai)'
]

for column in columns_to_clean:
    data[column] = data[column].apply(lambda x: remove_characters(x))

print('Describe: ', data.describe())
print('Info: ', data.info())
data = data.dropna()

plt.figure(figsize=(8, 6))
sns.countplot(x=data['Company Name'])
plt.title('Distribution of Company Name')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, fmt='.1f', cmap='coolwarm')
plt.title("Heat map")
plt.show()

list_price = ['Launched Price (Pakistan)', 'Launched Price (India)', 'Launched Price (China)', 'Launched Price (USA)',
              'Launched Price (Dubai)']
fig, axes = plt.subplots(2, 5, figsize=(20, 15))
for i, item in enumerate(list_price):
    sns.lineplot(x=data[item], y=data['Back Camera'], marker='o', ax=axes[0, i])
    sns.lineplot(x=data[item], y=data['Front Camera'], marker='o', color='red', ax=axes[1, i])
plt.show()

x = data.drop(['Company Name', 'Model Name', 'Mobile Weight', 'Launched Price (Pakistan)', 'Launched Price (India)',
               'Launched Price (China)', 'Launched Price (USA)', 'Launched Price (Dubai)'], axis=1)
y = data['Launched Price (India)']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

preprocessor = ColumnTransformer(transformers=[
    ('num_transformer', StandardScaler(),
     ['RAM', 'Front Camera', 'Back Camera', 'Battery Capacity', 'Screen Size', 'Launched Year']),
    ('cat_transformer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=0.1 ,lowercase=True), 'Processor')
], remainder='passthrough')

# x_train = preprocessor.fit_transform(x_train)
# x_test = preprocessor.transform(x_test)
# reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
# models,predictions = reg.fit(x_train, x_test, y_train, y_test)

parameter = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 15],
    'criterion': ['squared_error', 'absolute_error'],
}
reg = RandomForestRegressor(random_state=2)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reg',RandomForestRegressor(random_state=2, n_estimators=200)),
    # ('GridSearchCV', GridSearchCV(estimator=reg, param_grid=parameter, scoring='r2', cv=6, verbose=0))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# best_model = model.named_steps['model'].best_estimator_
# y_pred = best_model.predict(x_test)

print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('R2 Score: ', r2_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Price (y_test)")
plt.ylabel("Predicted Price (y_pred)")
plt.title("Actual vs Predicted Prices")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), sorted(y_train), label="y_train", linestyle="dashed", color="blue")
plt.plot(range(len(y_test)), sorted(y_test), label="y_test", linestyle="dashed", color="green")
plt.plot(range(len(y_pred)), sorted(y_pred), label="y_pred", linestyle="solid", color="red")
plt.xlabel("Index")
plt.ylabel("Price")
plt.title("Comparison of y_train, y_test, and y_pred")
plt.legend()
plt.show()