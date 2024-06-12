import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Set up the path to the dataset
train_dir = '/Users/andrewpoulin/Developer/ML-Dedect-QC/train/cropped'

left_pixels = 225
top_pixels = 198
right_pixels = 190
bottom_pixels = 1078
    


#test = crop("train/Class1/trigger_20240327_000000.844.png")


train_datagen = ImageDataGenerator(
    rescale=1./255,       
    shear_range=0.2,       
    zoom_range=0.2,        
    horizontal_flip=False,   
)


train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(374,374),  
    batch_size=32,
    class_mode='binary',

)
# plt.imshow(train_set[0][0][0])
# #result_data     = train_datagen.flow(train_set[0:5])

# image_list = []

# for images,labels in next(zip(train_set)):
#   for i in range(32): # can't be greater than 20
#     image_list.append(images[i])

# image_list = np.array(image_list)
# image_list.shape # (16,150,150,3)
# plt.imshow(image_list[3])


# Building the CNN
model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(374, 374, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Fully connected
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))  # Single output unit with sigmoid activation for binary classification

# Compiling
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=25
)

# Save the trained model
model.save('cnn_model_binary_test.h5')

print("Training completed")
