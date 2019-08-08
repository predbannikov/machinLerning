from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
import hashlib
import os
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    print(data.shape)
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:
            test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def test_my(identifier, test_check):
    return identifier < test_check

housing = load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
#housing["income_cat"] = np.ceil(housing["median_income"])
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
#print(type(housing["income_cat"]))
#print(housing["income_cat"])

housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(train_set), "train +", len(test_set), "test")

#fetch_housing_data()
#housing["ocean_proximity"].value_counts()

#print("I``````````````````````````")
#inc_pd = housing["income_cat"] 
#inc_pd2 = inc_pd.apply(lambda id_:
#    test_my(id_, 4))
#print(inc_pd2.head)
#inc_pd3 = inc_pd.loc[inc_pd2]
#print(len(inc_pd3))
#print("I______________")

#housing.info()
#housing.describe()
#housing.hist(bins=50, figsize=(20,10))
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
print(housing.info())
print(housing.head())
housing = strat_train_set.copy()
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

housing_cat = housing["ocean_proximity"]
print(housing_cat.head(10))
housing_cat_encoded, housing_categories = housing_cat.factorize()
print(housing_cat_encoded[:10])
print(housing_categories)
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_1hot.toarray())
#lin_reg = LinearRegression()
#lin_reg.fit(housing, housing_labels)
#some_data = housing.iloc[:5]
#some_labels = housing_labels.iloc[:5]
#some_data_prepared = full.pipeline.transform(some_data)
#print("Прогнозы:", list(some_labels))

print(housing.info())
print( housing.head())


#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, 
#        s=housing["population"]/100, label="population",
#        figsize=(10, 7), c="median_house_value",
#        cmap=plt.get_cmap("jet"), colorbar=True,)
housing["median_house_value"] = housing_labels
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.legend()
#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12,8))
plt.show()
