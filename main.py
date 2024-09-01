import pickle
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# dir = "C:\\Users\\TOSHIBA\\Desktop\\New folder (4)\\PetImages"
#
# categories = ["Cat","Dog"]
# data = []
#
# for category in categories:
#     path = os.path.join(dir,category)
#     label = categories.index(category)
#
#     for img in os.listdir(path):
#         imgpath = os.path.join(path,img)
#         petimg = cv2.imread(imgpath,0)
#         try:
#             petimg = cv2.resize(petimg,(50,50))
#             image = np.array(petimg).flatten()
#
#             data.append([image,label])
#         except Exception:
#             pass
#
# pick_in = open("data1.pickle",'wb')
# pickle.dump(data,pick_in)
# pick_in.close()


pick_in = open("data1.pickle", 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.25)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

pick = open("model1.sav", 'rb')
pickle.dump(model, pick)
pick.close()

prediction=model.predict(xtest)
accuracy = model.score(xtest,ytest)
#
categories = ["cat","dog"]
#
print("Accuracy:",accuracy)
print("Prediction is: ",categories[prediction[0]])
#
mypet = xtest[0].reshape(50,50)
plt.imshow(mypet, cmap="gray")
plt.show()
