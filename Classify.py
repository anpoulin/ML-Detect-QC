import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv

model = tf.keras.models.load_model('cnn_model_binary.h5')

image_folder = '/Users/andrewpoulin/Developer/ML-Dedect-QC/Classify91/'

target_size = (374,374)

# Function to preprocess and predict class of an image
def classify_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    print(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    print(prediction)
    
    class_label = 'class1' if prediction < 0.1 else 'class2'
    if class_label == 'class2':
        new_filename = img_path[:-4] + '_Class2.png'
        img.save(new_filename)
        print(f'Image saved as {new_filename}')
    return class_label, prediction

classification_results = []

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    print(img_name)
    if img_path.endswith(('png')):  
        class_label, prediction = classify_image(img_path)
        classification_results.append((img_name, class_label,prediction))


output_csv = 'classification_results.csv'

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Predicted Class'])  
    writer.writerows(classification_results)  

print(f"Classification results saved to {output_csv}")
