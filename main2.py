# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.preprocessing import image
import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix
  
  
img_width, img_height = 100, 100
  
train_data_dir = r'H:\Research_VIdeos\Research\Initial model\data\train_LPBD'
validation_data_dir = r'H:\Research_VIdeos\Research\Initial model\data\validation_LPBD'
img_load_test = r'H:\Research_VIdeos\Research\Initial model\data\validation_LPBD\.ঢাকা\segment 2 (9).jpg'

nb_train_samples = 1615
nb_validation_samples = 1615
epochs = 1
batch_size = 6
num_of_class = 12
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
  
model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape = input_shape))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 


model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =(2, 2)))
  
model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_of_class))
model.add(Activation('softmax'))
  
model.compile(loss ='categorical_crossentropy',
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 
  
train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2,
            horizontal_flip = True) 
  
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='categorical')
  
validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='categorical')

#model.load_weights('model_saved_2.h5')

H = model.fit_generator(train_generator,
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size) 


model.save_weights('model_saved_LPBD_no_pretrained.h5')


################################################prediction##############################
img_pred = image.load_img(img_load_test, target_size = (100, 100))
img_pred.show()
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)



result = model.predict(img_pred).tolist()
#result = result1.tolist()
print(type(result))
print("Dhaka ","Metro ", "0 ", "1 ", "2 ", "3 ", "4 ", "5 ","6 ","7 ","8 ","9 " )

if(result[0]==1):
    print("Dhaka")
elif(result[1]==1):
    print("Metro")
elif(result[2]==1):
    print("0")
elif(result[3]==1):
    print("1")
elif(result[4]==1):
    print("2")
elif(result[5]==1):
    print("3")
elif(result[6]==1):
    print("4")
elif(result[7]==1):
    print("5")
elif(result[8]==1):
    print("6")
elif(result[9]==1):
    print("7")
elif(result[10]==1):
    print("8")
elif(result[11]==1):
    print("9")



print(result)

'''print(H.history)

for key in H.history:
    print(key, H.history[key])'''

import matplotlib.pyplot as plt

#history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# Plot training & validation accuracy values
plt.plot(H.history["accuracy"])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



######### confusion matrix ################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
#target_names = ['Dhaka', 'Metro', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
target_names = ['1', '2']
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')
plt.show()
    


#print('Confusion Matrix')
#print(confusion_matrix(validation_generator.classes, y_pred))
#print('Classification Report')

#print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
#cm = confusion_matrix(validation_generator.classes, y_pred)
#cm_plot_labels = ['ss', 'sdd']
#plot_confusion_matrix(cm, cm_plot_labels, title = 'sdssdddd')