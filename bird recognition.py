import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load your model that you download
model_path = r'C:\Users\komsi\Desktop\Bird recognition\kers\converted_keras (3)\keras_model.h5'
model = tensorflow.keras.models.load_model(model_path)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image_path = r'C:\Users\komsi\Desktop\Bird recognition\to test\diamond_dove1.jpeg'
image = Image.open(image_path)

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
image.show()
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference[]
prediction = model.predict(data)
#in the label file you can find the classes or data you provided in the right order
#assign each class to it's place
cockatiel = prediction[0,0] 
diamond_dove = prediction[0,1]
Red_rumped_parrot = prediction[0,2]
yellow_fronted_canary = prediction[0,3]
print(prediction) # to print all the predictions as number without assigning names to it

#converting each prediction into a percent 
cockatiel_percentage = cockatiel*100 
diamond_dove_percentage = diamond_dove*100
Red_rumped_parrot_percentage = Red_rumped_parrot*100
yellow_fronted_canary_percentage = yellow_fronted_canary*100

#if any precent is bigger than 50 , it's most likely this bird

if cockatiel_percentage > 50:
    print('it is a cockatiel , with confidence of = ' , cockatiel_percentage)
elif diamond_dove_percentage > 50:
    print('it is a diamond dove , with confidence of = ' , diamond_dove_percentage)
elif Red_rumped_parrot_percentage > 50:
    print('it is a Red rumped parrot , with confidence of = ' , Red_rumped_parrot_percentage)
elif yellow_fronted_canary_percentage > 50:
    print('it is a yellow fronted canary , with confidence of = ' , yellow_fronted_canary_percentage)
else:
    print('i can not recognize this bird')
    
    