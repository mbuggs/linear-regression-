Python 3.11.1 (v3.11.1:a7a450f84a, Dec  6 2022, 15:24:06) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... from scipy import stats
... 
... 
... data = pd.read_csv('NFL2022.csv')
... 
... x = data['OverUnderLine']
... y = data['TotalScore']
... 
... 
... b1, b0, _, _, _ = stats.linregress(x, y)
... 
... 
... plt.figure(figsize=(8, 6))
... plt.scatter(x, y, label='Data', color='b')
... plt.plot(x, b0 + b1 * x, label='Fitted Line', color='r')
... plt.xlabel('Over/Under Line')
... plt.ylabel('Total Score')
... plt.title('NFL 2022 - Total Scores vs. Over/Under Line')
... plt.legend()
... plt.grid(True)
... plt.show()
... 
... 
... plt.figure(figsize=(8, 6))
... plt.hist([y, x], bins=20, label=['Total Scores', 'Over/Under Line'], color=['b', 'r'], alpha=0.6)
... plt.xlabel('Score/Line')
... plt.ylabel('Frequency')
... plt.title('Histogram of Total Scores and Over/Under Line')
... plt.legend()
... plt.grid(True)
... plt.show()
... 
... 
residuals = y - (b0 + b1 * x)


plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, density=True, label='Residuals', color='b', alpha=0.6)


mu, sigma = stats.norm.fit(residuals)
x_normal = np.linspace(min(residuals), max(residuals), 100)
y_normal = stats.norm.pdf(x_normal, mu, sigma)
plt.plot(x_normal, y_normal, 'r', label='Normal Curve')

plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Histogram of Residuals and Fitted Normal Curve')
plt.legend()
plt.grid(True)
plt.show()


std_errors = np.std(residuals)
print("Standard Deviation of Errors (Residuals):", std_errors)

