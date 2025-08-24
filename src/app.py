import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
matplotlib.use('TkAgg')  # Alternatif backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # Çıkan sonuçları görselleştirmek için

from sklearn.preprocessing import StandardScaler   # bu yöntemle standartize etme
from sklearn.model_selection import train_test_split, GridSearchCV
# GridSearchCrossValidation KNN ile ilgili en iyi parametreleri bulurken kullanıcaz
from sklearn.metrics import accuracy_score, confusion_matrix
# çıkan sonuçları değerlendirmek için doğruluk değeri kullanmamız lazım ve biz nerede ve nasıl hata yaptığımızı confusion matrix sayesinde bulabiliriz (neresi yanlış neresi doğru gibi şeylerde accuracy yetmiyor bu yüzden confisoun matrix kullanıyoruz)
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# hataları düzeltmemiz lazım ama warningler genelde sürümden kaynaklandığı için düzeltmemize gerek yok
import warnings
warnings.filterwarnings("ignore")

# pathler
from backbone import DATA_DIR

# veriyi okuma
data = pd.read_csv(DATA_DIR)

# veri setinde anlamsız olan bazı değerleri (satır, sütun) kaldırma
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# bazı (sütunların veya satırların kısacası) değerlerin sürekli kullanılacağı için akılda kalıcı bir isimle değiştirilmesi
data.rename(columns= {"diagnosis": "target"}, inplace=True)

# malptloit kütüphanesi ile gösterme yaılıyor
# sns.countplot(data["target"])
# plt.show()

# istenen değerin sayısının öğrenilmesi ve yorum yapılması
print(data.target.value_counts())

# ve kategorik değerleri sayısal değere döndürme hem matematiksel olarak hem de eğitim zamanında işe yarar
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]
# list compheransion açılımı

# target_list = []
# for i in data.target:
#     if i.strip() == "M":
#         target_list.append(1)
#     else:
#         target_list.append(0)
#
# data["target"] = target_list

print(len(data))
print(data.head())
print("Data shape", data.shape)

print(data.info())

describe = data.describe()
print(describe.to_string())

"""
Veriler arasında çok fazla fark var. Mesela bir sütundaki verilerin değeri çok fazla diğer sütundaki verilerin değeri de
çok az. O yüzden veriyi standardize etmemiz lazım.
missing value: None
"""

# TODO: EDA aşaması

# Correlation
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Matrix")

# istenilen herhangi bir özelliği ayrı bir matrix şekilde gösterebiliriz
threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")
# plt.show()

"""
there some correlation features
"""

# box plot
data_melted = pd.melt(data, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure()
sns.boxplot(x = "features", y = "value", hue = "target", data = data_melted)
plt.xticks(rotation = 90)
# plt.show()

"""
standardization-normalization
"""

#pair plot
sns.pairplot(data[corr_features], diag_kind= "kde", markers = "+", hue="target")
# plt.show()

"""
bu işlemelerle skewness'lara bakıyoruz
eğer varsa bunu düzeltmeniz lazım. 
if skew() > 1 ise pozitif, else negatif
"""

# TODO: Outlier verisetindeki aykırı değerlerdir. Eğer bunları düzenlenmezse modeli sapıtabilir
y = data.target
x = data.drop(["target"], axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

# threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()

"""
# örnek
features_plot = ["columnX","columnY"]
plt.scatter(x.loc[outlier_index,features_plot[0]],x.loc[outlier_index,features_plot[1]],...)
plt.scatter(x[features_plot[0]], x[features_plot[1]],...)

# veya şöyle
col1_idx = x.columns.get_loc("age")
col2_idx = x.columns.get_loc("income")
plt.scatter(x.iloc[outlier_index, col1_idx], x.iloc[outlier_index, col2_idx])
plt.scatter(x.iloc[:, col1_idx], x.iloc[:, col2_idx],...)
"""

# outlier gösterimi
plt.figure()
plt.scatter(x.iloc[outlier_index,1], x.iloc[outlier_index,2], color= "blue", s = 50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color= "k", s = 3, label = "Data Points")

radius = (x_score.max()-x_score) / (x_score.max()-x_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0], x.iloc[:,1], s = 1000*radius, edgecolor = "r", facecolors = "none" , label = "Data Points")
plt.legend()
# plt.show()

# drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

# %% Train test split
test_size = 0.3
"""
veri boyutu büyüdükçe test_size düşük seçilmelidir
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size =test_size, random_state=42)

# %%

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
print("new df:\n", X_train_df_describe)

X_train_df["target"] = Y_train

# box plot
data_melted = pd.melt(X_train_df, id_vars = "target",
                      var_name = "features",
                      value_name = "value")
plt.figure()
sns.boxplot(x = "features", y= "value", hue="target", data = data_melted)
plt.xticks(rotation = 90)
# plt.show()

# pair plot
sns.pairplot(X_train_df[corr_features], diag_kind="kde", markers = "+", hue="target")
# plt.show()



knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
y_predict = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_predict)
acc = accuracy_score(Y_test, y_predict)
score = knn.score(X_test, Y_test)
print("Score: ", score)
print("CM:", cm)
print("Basic KNN Acc: ", score)


def KNN_Best_Params(x_train, x_test, y_train, y_test):

    k_range = list(range(5,31))
    weight_options = ["uniform", "distance"]
    metric_options = ['euclidean', 'manhattan', 'chebyshev']
    print("\n")
    param_grid = dict(n_neighbors = k_range, weights = weight_options, metric = metric_options)

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')
    grid.fit(x_train, y_train)

    print("Best training score: {} with parameters: {}".format(grid.best_score_, grid.best_params_))
    print("\n")

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)

    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)

    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score: {}\n".format(acc_test, acc_train))

    print("Confusion matrix:\n", cm_test)
    print("CM Train:\n", cm_train)

    return grid

grid = KNN_Best_Params(X_train, X_test, Y_train, Y_test)


# %% PCA

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
pca.fit(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca, columns = ["p1", "p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1", y = "p2", hue = "target", data=pca_data)
plt.title("PCA: p1 vs p2")


X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca, y, test_size=test_size, random_state= 42)

grid_pca = KNN_Best_Params(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)

# visulize
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .05  # step size in the mesh
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() +1
xx, yy = np.meshgrid((np.arange(x_min, x_max, h)),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap= cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)), grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))




# %% NCA

nca = NeighborhoodComponentsAnalysis(n_components=2, random_state= 42)
nca.fit(x_scaled, y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1", "p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1", y = "p2", hue = "target", data= nca_data)
plt.title("NCA: p1 vs p2")

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size=test_size, random_state= 42)

grid_nca = KNN_Best_Params(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)


# visulize
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2  # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() +1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() +1
xx, yy = np.meshgrid((np.arange(x_min, x_max, h)),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap= cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)), grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))


plt.show()











