import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dropout, Flatten, Dense  
from tensorflow.keras import applications  
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt  
import math  
import cv2  
import datetime
from tkinter import filedialog


# dimensions of our images.  
img_width, img_height = 224, 224  
   
top_model_weights_path = 'bottle neck_fc_model.h5'
train_data_dir = 'datasets'  
validation_data_dir = 'datasets'  
test_data_dir='images'
# number of epochs to train top model

epochs = int(input("batch : "))
print("batch size : ",epochs )
# batch size used by flow_from_directory and predict_generator
b_size = 10 #epoch
print("epochs : ",b_size,"\n\n" )


#Loading vgc16 model
vgc_16 = applications.VGG16(include_top=False, weights='imagenet')

start = datetime.datetime.now()
datagen = ImageDataGenerator(rescale=1. / 255)  
   
generator = datagen.flow_from_directory(  
     train_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=b_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_train_samples = len(generator.filenames)  
num_classes = len(generator.class_indices)
 
   
predict_size_train = int(math.ceil(nb_train_samples / b_size))  
   
bottleneck_features_train = vgc_16.predict_generator(generator, predict_size_train)  
   
np.save('bottleneck_features_train.npy', bottleneck_features_train)

# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  
   
# get the class labels for the training data, in the original order  
train_labels = generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)

end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     validation_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=b_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_validation_samples = len(generator.filenames)  
   
predict_size_validation = int(math.ceil(nb_validation_samples / b_size))  
   
bottleneck_features_validation = vgc_16.predict_generator(  
     generator, predict_size_validation)  
   
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

validation_data = np.load('bottleneck_features_validation.npy')  



validation_labels = generator.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

start = datetime.datetime.now()
generator = datagen.flow_from_directory(  
     test_data_dir,  
     target_size=(img_width, img_height),  
     batch_size=b_size,  
     class_mode=None,  
     shuffle=False)  
   
nb_test_samples = len(generator.filenames)
test_data = np.load('bottleneck_features_train.npy')
test_labels = generator.classes  
test_labels = to_categorical(test_labels, num_classes=num_classes)
   
predict_size_test = int(math.ceil(nb_test_samples / b_size))  
   
bottleneck_features_test = vgc_16.predict_generator(  
     generator, predict_size_test)  
   
np.save('bottleneck_features_test.npy', bottleneck_features_test) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)

filename = filedialog.askopenfilename(title='"pen')
#print(filename)
#raw_image = cv2.imread(filename)
#cv2.imshow('img', raw_image)
image_path = filename

orig = cv2.imread(image_path)  

print("[INFO] loading and preprocessing image...")  
image = load_img(image_path, target_size=(224, 224))
img= load_img(image_path, target_size=(224, 224)) 
image = img_to_array(image)  

# important! otherwise the predictions will be '0'  
image = image / 225

image = np.expand_dims(image, axis=0)  
# print(image)

# build the VGG16 network  
model = applications.VGG16(include_top=False, weights='imagenet')
model.summary()


# get the bottleneck prediction from the pre-trained VGG16 model  
bottleneck_prediction = model.predict(image)  

# build top model  
model = Sequential()  
model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))  
model.add(Dense(256, activation='relu'))  
#model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer="rmsprop",metrics=['acc'])  
#model.load_weights(top_model_weights_path)  

history = model.fit(train_data, train_labels,
      epochs,
      b_size,  
      validation_data=(validation_data, validation_labels))  

model.save_weights(top_model_weights_path)  

(eval_loss, eval_accuracy) = model.evaluate(  
 validation_data, validation_labels, b_size, verbose=1)

# use the bottleneck prediction on the top model to get the final classification  
class_predicted = model.predict_classes(bottleneck_prediction)

inID = class_predicted[0]  

class_dictionary = generator.class_indices  

inv_map = {v: k for k, v in class_dictionary.items()}  

label = inv_map[inID]  

# get the prediction label  
print("Label: {}".format(label))  
print("Accuracy: {}",history.history['acc'][3])
import scipy.ndimage as ndi
import functools
b = np.array(([0,1,1,0],
                  [0,1,0,0],
                  [0,0,0,0],
                  [0,0,1,1],
                  [0,0,1,1]))


cy, cx = ndi.center_of_mass(b)

plt.figure()
plt.imshow(img, cmap='gray')  
plt.scatter(cx, cy)# show me its center
#plt.title('Confusion Matrix:')
#plt.show()
bwimshow = functools.partial(plt.imshow, vmin=0, vmax=255,
                             cmap=plt.get_cmap('gray'))
dots = np.random.randn(10, 10)*255
bwimshow(dots)
cbar = plt.colorbar()
###########################plt.title('Confustion Matrix')
###################################plt.show()
#
#from sklearn.metrics import accuracy_score
#accuracy_score(labels_test,pred)
print(accuracy_score)

print(history.history['acc'][3])
plt.figure(1)  

# summarize history for accuracy  

plt.subplot(211)

plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  

# summarize history for loss  

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
