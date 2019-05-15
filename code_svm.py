
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
train_path = "/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/train.csv"
# test_path = "/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/test.csv"
#test_path = "/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/test_veri.csv"
# test_path = "/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/testing/testing1.csv"
test_path = "/Users/yogesh/Desktop/CollegeProject/CV_proj/cv_oct/testing/testing1.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

label_train=train_data['label']
train=train_data.drop('label', axis=1)

train1=train.astype('float16')
train1=(train1)/255
test1=test_data.astype('float16')/255

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train1, label_train, train_size = 0.8,random_state = 42)

from sklearn import decomposition

pca = decomposition.PCA(n_components=100)
pca.fit(X_train)

PCtrain = pca.transform(X_train)
PCval = pca.transform(X_val)

PCtest = pca.transform(test1)

X_train= PCtrain

X_cv = PCval
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train,y_train)

predicted = clf.predict(X_cv)
expected = y_val
import seaborn as sns
output_label = clf.predict(PCtest)

output = pd.DataFrame(output_label,columns = ['Label'])
output.reset_index(inplace=True)
output['index'] = output['index'] + 1
output.rename(columns={'index': 'ImageId'}, inplace=True)
output.to_csv('output.csv', index=False)



# confusion_mtx = confusion_matrix(expected, predicted)
# # plot the confusion matrix
# f,ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()
