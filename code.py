# Lara Foqaha 1220071 Section 4 || Dana Taher 1221240 Section 1

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

############# loading and processing images ###########
dataset_location = r"C:\Users\HP\Downloads\dataset"
resize_to = (32, 32)

X = [] #image data
y = [] # labels
label = {} # maps label index to class name

# loop folders in dataset
for i, class_name in enumerate(sorted(os.listdir(dataset_location))):
    label[i] = class_name # assigning label index to class name
    class_location = os.path.join(dataset_location, class_name)
    for file in os.listdir(class_location):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = os.path.join(class_location, file)
                image = Image.open(image_path).convert("RGB")
                image = image.resize(resize_to)
                arr = np.array(image, dtype=np.float32) / 255.0 # normalizing pixel values
                X.append(arr.flatten())
                y.append(i)
            except Exception as e:
                print(f"Skipped {file}: {e}")

X = np.array(X)
y = np.array(y)

# splitting data to train/test 
# test 25% and train 75%
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42, stratify=y)

# training naive bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# training decision tree model
dt = DecisionTreeClassifier(
    criterion='entropy', # entropy for splitting      
    max_depth=None,   # growing fully
    min_samples_split=2,         
    min_samples_leaf=1,         
    random_state=42
)
dt.fit(X_train, y_train)

# training MLP neural network model
mlp = MLPClassifier(
    hidden_layer_sizes=(128,64), # two hidden layers, one with 128 neurons the other with 64
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
mlp.fit(X_train, y_train)


########### evaluation function ##########
def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test) # predicting the class for images in test set
    acc = accuracy_score(y_test, y_pred) # calculating accuracy

    print("\n" + "============================")
    print(f"{name} Results")
    print("============================")
    print(f"Accuracy: {acc:.4f}\n")

    names = [] #list of class names
    for i in sorted(label.keys()):
        names.append(label[i])

    # precision, recall, f1 for each class
    report = classification_report( y_test, y_pred, output_dict=True, target_names=names)

    for i in sorted(label.keys()):
        class_name = label[i]
        print(f"Class: {class_name}")
        print(f"Precision: {report[class_name]['precision']:.2f}")
        print(f"Recall: {report[class_name]['recall']:.2f}")
        print(f"F1-score: {report[class_name]['f1-score']:.2f}\n")

##### confusion matrix #######
def show_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

labels_nb = []
for i in sorted(label):
    labels_nb.append(label[i])
show_confusion_matrix(y_test, nb.predict(X_test), labels_nb, "Naive Bayes")

labels_dt = []
for i in sorted(label):
    labels_dt.append(label[i])
show_confusion_matrix(y_test, dt.predict(X_test), labels_dt, "Decision Tree")

labels_mlp = []
for i in sorted(label):
    labels_mlp.append(label[i])
show_confusion_matrix(y_test, mlp.predict(X_test), labels_mlp, "MLP Neural Net")

######### visual decision tree #########
plt.figure(figsize=(20, 10))

class_names = []
for i in sorted(label):
    class_names.append(label[i])

plot_tree(
    dt,
    feature_names=None,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=None # fully grow
    )

plt.title("Decision Tree Visualization")
plt.show()

############################
print("\nAccuracies:\n")
print(f"Naive Bayes: {accuracy_score(y_test, nb.predict(X_test)):.4f}")
print(f"Decision Tree: {accuracy_score(y_test, dt.predict(X_test)):.4f}")
print(f"MLP Neural Net: {accuracy_score(y_test, mlp.predict(X_test)):.4f}")

evaluate(nb, "Naive Bayes", X_test, y_test)
evaluate(dt, "Decision Tree", X_test, y_test)
evaluate(mlp, "MLP Neural Net", X_test, y_test)