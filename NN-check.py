import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train , y_train) , (x_test , y_test) = mnist.load_data()
#normalize matrixes 0 -> 1
x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)

#make model to load images
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128 , activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10 , activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train , y_train , epochs = 3)
validation_loss , val_acc = model.evaluate(x_test , y_test)
print(validation_loss, val_acc)

#model.sxave('epic_num_reader.model')
#loaded_model = tf.keras.models.load_model('epic_num_reader.model')

'''prediction = model.predict([x_test])
import numpy as np
print(np.argmax(prediction[0]))

plt.imshow(x_test[0])
plt.show()


import matplotlib.pyplot as plt
plt.imshow(x_train[0] , cmap = plt.cm.binary)
plt.show()'''


