<div>
<h1>Linear Regression Tutorial</h1>
</div>

<div>
<img src="thumbnail.png" width="800">
</div>

# Import All necessary library


```python
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
```

# Read the Dataset


```python
df = pd.read_csv('car data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Car_Name</th>
      <th>Year</th>
      <th>Selling_Price</th>
      <th>Present_Price</th>
      <th>Kms_Driven</th>
      <th>Fuel_Type</th>
      <th>Seller_Type</th>
      <th>Transmission</th>
      <th>Owner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ritz</td>
      <td>2014</td>
      <td>3.35</td>
      <td>5.59</td>
      <td>27000</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>sx4</td>
      <td>2013</td>
      <td>4.75</td>
      <td>9.54</td>
      <td>43000</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ciaz</td>
      <td>2017</td>
      <td>7.25</td>
      <td>9.85</td>
      <td>6900</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>wagon r</td>
      <td>2011</td>
      <td>2.85</td>
      <td>4.15</td>
      <td>5200</td>
      <td>Petrol</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>swift</td>
      <td>2014</td>
      <td>4.60</td>
      <td>6.87</td>
      <td>42450</td>
      <td>Diesel</td>
      <td>Dealer</td>
      <td>Manual</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Data Exploration


```python
sn.heatmap(df.corr(), annot=True)
```




    <AxesSubplot:>




![png](Complete%20Code_files/Complete%20Code_6_1.png)



```python
plt.scatter(df['Selling_Price'],df['Present_Price'],marker = '.')

# Selling_Price: This is the price the owner wants to sell the car at.
# Present_Price: This is the current ex-showroom price of the car.
```




    <matplotlib.collections.PathCollection at 0x7fabcf81dcd0>




![png](Complete%20Code_files/Complete%20Code_7_1.png)



```python
plt.scatter(df['Kms_Driven'],df['Present_Price'],marker = '.')
```




    <matplotlib.collections.PathCollection at 0x7fabcf9a64d0>




![png](Complete%20Code_files/Complete%20Code_8_1.png)


# Model Functions


```python
x = df['Selling_Price'].to_numpy()
y = df['Present_Price'].to_numpy()
def mean(values):
    return sum(values) / float(len(values))

# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
covar = covariance(x, mean_x, y, mean_y)

print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
print('Covariance: %.3f' % (covar))


# B1 = sum((x(i) - mean(x)) * (y(i) - mean(y))) / sum( (x(i) - mean(x))^2 )
# B0 = mean(y) - B1 * mean(x)
```

    x stats: mean=4.661 variance=7750.492
    y stats: mean=7.628 variance=22416.219
    Covariance: 11585.801


# Fit Model


```python
def coefficients(x,y):
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

b0,b1 = coefficients(x,y)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
```

    Coefficients: B0=0.661, B1=1.495


# Visualiza the Result


```python
predictions = []
for x_val in x:
    pred =  b0 + b1*x_val
    predictions.append(pred)
plt.scatter(x,y,marker = '.')
plt.plot(x,predictions)
```




    [<matplotlib.lines.Line2D at 0x7fabcfc71290>]




![png](Complete%20Code_files/Complete%20Code_14_1.png)



```python
plt.scatter(x,y,marker = '.')
plt.plot(x,predictions)
```




    [<matplotlib.lines.Line2D at 0x7fabcfd31950>]




![png](Complete%20Code_files/Complete%20Code_15_1.png)



```python

```
