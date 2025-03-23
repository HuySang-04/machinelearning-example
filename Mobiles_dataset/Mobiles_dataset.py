#https://www.kaggle.com/datasets/abdulmalik1518/mobiles-dataset-2025
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/sang/BaiTap/MachineLearning/Mobiles_dataset/Mobiles Dataset (2025).csv', encoding='latin1')

def remove_characters(value):
    if pd.isna(value) or value == '' or value == ' ':
        return np.nan
    str_number = re.findall(r'[\d,.]+', str(value))
    if len(str_number) == 0:
        return np.nan
    try:
        cleaned_numbers = [float(num.replace(',', '.')) for num in str_number]
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

plt.figure(figsize=(8, 6))
sns.countplot(x=data['Company Name'])
plt.title('Distribution of Company Name')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, fmt='.1f', cmap='coolwarm')
plt.title("Heat map")
plt.show()

list_price = [
    'Launched Price (Pakistan)', 'Launched Price (India)',
    'Launched Price (China)', 'Launched Price (USA)',
    'Launched Price (Dubai)'
]

fig, axes = plt.subplots(2, 5, figsize=(20, 15))
for i, item in enumerate(list_price):
    sns.lineplot(x=data[item], y=data['Back Camera'], marker='o', ax=axes[0, i])
    sns.lineplot(x=data[item], y=data['Front Camera'], marker='o', color='red', ax=axes[1, i])
plt.show()

print(data['Launched Price (USA)'].max())