from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = pad_sequences(data_dict['data'], dtype=object, padding='post', truncating='post')

labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

table = PrettyTable()
table.field_names = ['Metric', 'Score']
table.add_row(['Accuracy', f'{accuracy:.2%}'])
table.add_row(['Precision', f'{precision:.2%}'])
table.add_row(['Recall', f'{recall:.2%}'])
table.add_row(['F1-Score', f'{f1:.2%}'])

print(table)

conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
