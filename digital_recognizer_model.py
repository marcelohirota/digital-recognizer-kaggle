# Digital Recognizer Model

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# Loading datasets - train and test

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X = df.iloc[:,1:].values
y = df.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Creating training model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

# Fitting the model
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Creating the predictions for submission
y_subm = classifier.predict(df_test)

img = df_test.iloc[0, :].values
img = img.reshape(28,28).astype('uint8')
plt.imshow(img)

# Accuracy of model
count=0
for i in range (len(y_pred)):
    if y_pred[i]==y_test[i]:
        count= count+1
accuracy = count/len(y_pred)
print(accuracy)

imageid = []
i=0
for i in range (len(y_subm)+1):
    imageid.append(i)  
imageid.pop(0)

# Creating submission file
submission = pd.DataFrame({
        "ImageId": imageid,
        'Label': y_subm
        })

submission.to_csv('submission_drver1.csv', index = False)