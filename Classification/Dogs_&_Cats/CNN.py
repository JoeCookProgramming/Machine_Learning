import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import random
import shutil
import sys

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# Get the folder with the training data
directory = os.path.dirname(__file__)
trainingfolder = os.path.abspath(os.path.join(directory, '..', 'train'))

def imagePlot(catdog,folder):
	for i in range(9):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# define filename
		filename = os.path.abspath(os.path.join(folder,f'{catdog}.{i}.jpg'))
		# load image pixels
		image = imread(filename)
		# plot raw pixel data
		pyplot.imshow(image)
	# show the figure
	pyplot.show()

# Not necessary but can speed up the process, not currently being used
def resizeImages(folder):
	photos, labels = list(), list()
	for file in listdir(folder):
		# determine class
		output = 0.0
		if file.startswith('cat'):
			output = 1.0
		# load image
		photo = load_img(os.path.abspath(os.path.join(folder,file)), target_size=(200, 200))
		# convert to numpy array
		photo = img_to_array(photo)
		# store
		photos.append(photo)
		labels.append(output)
	# convert to a numpy arrays
	photos = asarray(photos)
	labels = asarray(labels)
	# save the reshaped photos
	save('dogs_vs_cats_photos.npy', photos)
	save('dogs_vs_cats_labels.npy', labels)
	print(photos.shape, labels.shape)


def createDirectories(dataset_home,subdirs,labeldirs):
	for subdir in subdirs:
		# create label subdirectories
		for labldir in labeldirs:
			newdir = dataset_home + subdir + labldir
			os.makedirs(newdir, exist_ok=True)


def populateDirectories(trainingfolder,dataset_home):
	# seed random number generator
	random.seed(1)
	# define ratio of pictures to use for validation
	val_ratio = 0.25
	# copy training dataset images into subdirectories
	for file in listdir(trainingfolder):
		src = trainingfolder + '/' + file
		dst_dir = 'train/'
		if random.random() < val_ratio:
			dst_dir = 'test/'
		if file.startswith('cat'):
			dst = dataset_home + dst_dir + 'cats/'  + file
			shutil.copyfile(src, dst)
		elif file.startswith('dog'):
			dst = dataset_home + dst_dir + 'dogs/'  + file
			shutil.copyfile(src, dst)


# define cnn model
def define_model(image_size):
	model = Sequential()
	#1 at the end for greyscale, 3 for rgb
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_size, image_size, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


# def run_test_harness():
#     # define model
# 	model = define_model()
# 	# create data generator
# 	train_datagen = ImageDataGenerator(rescale=1.0/255.0,
# 		width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# 	test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# 	# prepare iterators
# 	train_it = train_datagen.flow_from_directory(os.path.abspath(os.path.join(directory, 'dataset', 'train')),
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
# 	test_it = test_datagen.flow_from_directory(os.path.abspath(os.path.join(directory, 'dataset', 'test')),
# 		class_mode='binary', batch_size=64, target_size=(200, 200))
# 	# fit model
# 	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
# 		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
# 	# evaluate model
# 	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
# 	print('> %.3f' % (acc * 100.0))
# 	# learning curves
# 	summarize_diagnostics(history)


def run_test_harness(image_size):
    # define model
    model = define_model(image_size)
    # create data generator
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = train_datagen.flow_from_directory(os.path.abspath(os.path.join(directory, 'dataset', 'train')),
        color_mode= 'grayscale', class_mode='categorical', batch_size=128, target_size=(image_size, image_size))
    test_it = test_datagen.flow_from_directory(os.path.abspath(os.path.join(directory, 'dataset', 'test')),
        color_mode= 'grayscale', class_mode='categorical', batch_size=128, target_size=(image_size, image_size))
    # fit model

    model.fit_generator(
        train_it, 
        steps_per_epoch=len(train_it),
        validation_data=test_it, 
        validation_steps=len(test_it), 
        epochs=40,
        verbose=1)

    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    model.save('final_model.h5')

# load and prepare the image
def load_image(filename,image_size):
	# load the image
	img = load_img(filename, target_size=(image_size, image_size),color_mode="grayscale")
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	# 1 at the end for greyscale, 3 for rgb
	img = img.reshape(1, image_size, image_size, 1)
	# center pixel data
	img = img.astype('float32')
	# img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example(image_file, image_size):
	# load the image
	img = load_image(image_file, image_size)
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict_proba(img)
	print(f"Cat - {round(result[0][0] * 100,2)}%")
	print(f"Dog - {round(result[0][1] * 100,2)}%")


image_size = 200

# imageplot("cat",trainingfolder)
# resizeImages(trainingfolder)
#dataset_home = 'dataset/'
# subdirs = ['train/', 'test/']
# labeldirs = ['dogs/', 'cats/']
# createDirectories(dataset_home,subdirs,labeldirs)
#populateDirectories(trainingfolder,dataset_home)

#run_test_harness(image_size)

run_example('tiger.jpg',image_size)




