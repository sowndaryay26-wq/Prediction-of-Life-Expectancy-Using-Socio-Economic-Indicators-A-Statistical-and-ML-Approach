import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Life Expectancy Data.csv")

df.head()
df.info()
df.describe()

df = df.dropna()

plt.figure(figsize=(4, 10)) 

heatmap_data = df.corr(numeric_only=True)[['Life expectancy ']].sort_values(by='Life expectancy ', ascending=False)

sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Features Correlating with Life Expectancy")
plt.show()

sns.scatterplot(x='GDP', y='Life expectancy ', data=df)
plt.show()

X = df[['GDP', 'Schooling', 'Adult Mortality', ' BMI ', 'Alcohol']]
y = df['Life expectancy ']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("R2:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("RF R2:", r2_score(y_test, y_pred_rf))
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

importance = rf.feature_importances_
pd.Series(importance, index=X.columns).sort_values().plot(kind='barh')
plt.show()
