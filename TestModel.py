import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Set up the path to the test dataset
test_dir = '/Users/andrewpoulin/Developer/ML-Dedect-QC/test/'

# Load the trained model
model = load_model('cnn_model_binary.h5')

# ImageDataGenerator for preprocessing test data
test_datagen = ImageDataGenerator(rescale=1./255,)  # Rescale the images by a factor of 1/255

# Creating the test set
test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(374,374),  # Resize images to 64x64
    batch_size=32,
    class_mode='binary'    # Use 'binary' for binary classification
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_set, steps=test_set.samples // test_set.batch_size)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
