import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler


training_set = pd.read_excel("Final_Train.xlsx")
test_set = pd.read_excel("Final_Test.xlsx")

#filling NaN values
training_set.drop(['Rating'],axis=1,inplace=True)
training_set['Place'].fillna(training_set['Place'].value_counts().index[0],inplace=True)

test_set.drop(['Rating'],axis=1,inplace=True)
test_set['Place'].fillna(test_set['Place'].value_counts().index[0],inplace=True)


X_train = training_set.iloc[:,0:5].values
print(pd.DataFrame(X_train).head(10))


Y_train = training_set.iloc[:,-1].values
print(pd.DataFrame(Y_train).head(10))


X_test = test_set.iloc[:,0:5].values
print(pd.DataFrame(X_test).head(10))



print("==================Training set======================")
print(training_set.info(),"\n")
print(training_set.columns,"\n")
print(training_set.isnull().sum(),"\n")

print(training_set.head(),"\n")
#print(training_set['Qualification'].value_counts(),"\n")
print(training_set['Miscellaneous_Info'].value_counts().index[0],"\n")
print(training_set['Place'].value_counts().index[0],"\n")
print("===================Test set========================","\n")
print(test_set.info(),"\n")
print(test_set.columns,"\n")
print(test_set.isnull().sum(),"\n")


le_X = LabelEncoder()
OHE_X = OneHotEncoder()
binary_encoder = ce.BinaryEncoder()


#========BinaryEncoding=============================
df_encoded_X_train = binary_encoder.fit_transform(pd.DataFrame(X_train))
print(df_encoded_X_train.columns,"\n")
X_train_Binary = df_encoded_X_train.iloc[:,0:5].values

df_encoded_X_test = binary_encoder.fit_transform(pd.DataFrame(X_test))
print(df_encoded_X_test.columns,"\n")
X_test_Binary = df_encoded_X_test.iloc[:,0:5].values

print(X_test_Binary)
print(X_train_Binary)
#==================================================================
Y_train = Y_train.reshape(-1,1)


scaler = StandardScaler()
scaled_X_train_binary = scaler.fit_transform(X_train_Binary)
scaled_X_test_binary = scaler.fit_transform(X_test_Binary)
scaled_Y_train = scaler.fit_transform(Y_train)
print(scaled_X_test_binary,"\n\n",scaled_X_train_binary,"\n\n",scaled_Y_train)




#Initializing the Deision Tree Regressor
dtr = DecisionTreeRegressor()

# Fitting the Decision Tree Regressor with training Data
dtr.fit(scaled_X_train_binary,scaled_Y_train.flatten())

# Predicting the values(Fees) for Test Data
predicted_value = list()

Y_pred_dtr = scaler.inverse_transform(dtr.predict(X_test_Binary))

for i in range(len(X_test_Binary)):
    fee_values = int(Y_pred_dtr[i])
    predicted_value.append(fee_values)

print(predicted_value[0:500])


#pd.DataFrame(predicted_value, columns = ['Fees']).to_excel("fee.xlsx", index = False)
