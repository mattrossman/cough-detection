from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from features import load_features
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128, 87, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


x, y = load_features('data/features-large.npz')

x = x.reshape(-1, 128, 87, 1)

history = model.fit(x, y, validation_split=0.2, epochs=100, shuffle=True, verbose=1)
x_test = history.validation_data[0]
y_test = history.validation_data[1]
y_pred = model.predict_classes(x_test)

print(classification_report(y_test, y_pred))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
