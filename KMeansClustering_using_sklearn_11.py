import pandas as pd
import matplotlib.pyplot as plt

# appling the datasets:
df = pd.read_csv('income.csv')
print(df.head())

# ploting the graph using the scatter plot:
plt.scatter(df['Age'], df['Income($)'])
plt.show()

# K-means clustering:
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
print(y_predicted)

# adding the y_predicting column into the df dataframe:
df['cluster'] = y_predicted
print(df.head())

# separating the clusters:
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
# plotting the graph:
plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

# Minimize and Maximize the Age and Income using the MinMaxScaler:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# here we are doing on Income:
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
# here we are doing on Age:
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
print(df)

# Again we are doing the KMeans clustering for better performance:
km_1 = KMeans(n_clusters=3)
y_predicted_1 = km_1.fit_predict(df[['Age', 'Income($)']])
# print(y_predicted_1)

# updating the cluter columns for new clusters:
df['cluster'] = y_predicted_1
print(df)

# centroids of the different cluster:
print(km_1.cluster_centers_)

# now again we created an plot for the new updated clusters:
# separating the clusters:
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# plotting the graph:
plt.scatter(df1.Age, df1['Income($)'], color='green')
plt.scatter(df2.Age, df2['Income($)'], color='red')
plt.scatter(df3.Age, df3['Income($)'], color='black')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.scatter(km_1.cluster_centers_[:,0], km_1.cluster_centers_[:,1],color='purple', marker='+', label='centroid')
plt.show()

# plot the all clustering range from 1 to 10:
k_range = range(1, 10)
sse = []
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)

print(sse)

# plotting the clustering range on the graph:
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_range, sse)
plt.show()