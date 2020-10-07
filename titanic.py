import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# importing training/test data sets
train_data = pd.read_csv("train.csv", index_col="PassengerId")
test_data = pd.read_csv("test.csv", index_col="PassengerId")

# filling in men's/boy's missing ages
train_boy_filter = train_data["Name"].str.contains("Master.", regex=False)
train_age_na_filter = train_data["Age"].isna()
boys = train_data[train_boy_filter]
train_data.loc[train_boy_filter & train_age_na_filter, "Age"] = boys["Age"].mean()
train_data.loc[(train_data["Sex"] == "male") & train_age_na_filter, "Age"] = train_data["Age"].mean()

# filling in women's/girl's missing ages
train_parents_filter = train_data["Parch"] > 0
girls = train_data[train_parents_filter & (train_data["Sex"] == "female")]
train_data.loc[train_parents_filter &
               (train_data["Sex"] == "female") &
               train_age_na_filter, "Age"] = girls["Age"].mean()
train_data.loc[(train_data["Sex"] == "female") &
               train_age_na_filter, "Age"] = train_data["Age"].mean()

# filling in embarked NaNs
train_embarked_mode = train_data["Embarked"].mode()
train_data.fillna({"Embarked": train_embarked_mode[0]}, inplace=True)


# creating dummy variables for sex and embarked
train_dummies = pd.get_dummies(train_data.loc[:, ["Sex", "Embarked"]])
train_data = pd.concat([train_data, train_dummies], axis=1)


# dropping unnecessary columns
train_data.drop(["Name", "Sex", "Embarked", "Ticket", "Cabin"], axis=1, inplace=True)

# creating input data for neural network
train_X = train_data.loc[:, "Pclass":]
train_y = train_data.Survived


# CLEANING TEST DATA
# filling in men's/boy's missing ages
test_boy_filter = test_data["Name"].str.contains("Master.", regex=False)
test_age_na_filter = test_data["Age"].isna()
boys = test_data[test_boy_filter]
test_data.loc[test_boy_filter & test_age_na_filter, "Age"] = boys["Age"].mean()
test_data.loc[(test_data["Sex"] == "male") &
              test_age_na_filter, "Age"] = test_data["Age"].mean()

# filling in women's/girl's missing ages
test_parents_filter = test_data["Parch"] > 0
girls = test_data[test_parents_filter & (test_data["Sex"] == "female")]
test_data.loc[(test_parents_filter & (test_data["Sex"] == "female") &
               test_age_na_filter), "Age"] = girls["Age"].mean()
test_data.loc[(test_data["Sex"] == "female") &
              test_age_na_filter, "Age"] = test_data["Age"].mean()

# filling in fare NaNs
test_fare_mean = test_data["Fare"].mean()
test_data.fillna({"Fare": test_fare_mean}, inplace=True)

# creating dummy variables for sex and embarked
test_dummies = pd.get_dummies(test_data.loc[:, ["Sex", "Embarked"]])
test_data = pd.concat([test_data, test_dummies], axis=1)

# dropping unnecessary columns
test_data.drop(["Name", "Sex", "Embarked", "Ticket", "Cabin"], axis=1, inplace=True)


forest_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
forest_model.fit(train_X, train_y)
predictions = forest_model.predict(test_data)
predictions = np.rint(predictions)
predictions = predictions.astype(int)

print(predictions)

np.savetxt("output.csv", predictions, delimiter=",", header="Survived")
