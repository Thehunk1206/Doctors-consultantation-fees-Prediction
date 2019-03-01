import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


training_set = pd.read_excel("Final_Train.xlsx")
test_set = pd.read_excel("Final_Test.xlsx")
X_train = training_set.iloc[:,0:6].values
Y_train = training_set.iloc[:,-1].values
X_test = test_set.iloc[:,0:6].values

print(training_set.info(),"\n")
print(training_set.columns,"\n")
print(training_set.isnull().sum(),"\n")
print(training_set['Profile'].value_counts(),"\n")


binary_encoder = ce.BinaryEncoder()



df_encoded_X_train = binary_encoder.fit_transform(pd.DataFrame(X_train))
print(df_encoded_X_train.columns,"\n")
X_train_Binary = df_encoded_X_train.iloc[:,0:6].values


df_encoded_X_test = binary_encoder.fit_transform(pd.DataFrame(X_test))
print(df_encoded_X_test.columns,"\n")
X_test_Binary = df_encoded_X_test.iloc[:,0:6].values


print(X_test_Binary)
print(X_train_Binary)





#Initializing the Deision Tree Regressor
dtr = DecisionTreeRegressor()

# Fitting the Decision Tree Regressor with training Data
dtr.fit(X_train_Binary,Y_train)

# Predicting the values(Fees) for Test Data
predicted_value = list()

Y_pred_dtr = dtr.predict(X_test_Binary)

for i in range(len(X_test_Binary)):
    fee_values = int(Y_pred_dtr[i])
    predicted_value.append(fee_values)

print(predicted_value[0:500])


pd.DataFrame(predicted_value, columns = ['Fees']).to_excel("Prediction_fee.xlsx", index = False)
