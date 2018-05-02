import os
import shutil

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments.               #
# Users could set them from the project setting page.             #
###################################################################


input_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/train"
test_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/test"
validation_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/validation"
root_visualizaion_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/visualization"


#################################################################################
# Keras imports.                                                                #
#################################################################################
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224,224
train_data_dir = input_dir
validation_data_dir = validation_dir
nb_train_samples = 1672
nb_validation_samples = 560
train_steps = 100 # 1672 training samples/batch size of 32 = 52 steps. We are doing heavy data processing so put 500 here
validation_steps = 20 # 560 validation samples/batch size of 32 = 10 steps. We put 20 for validation steps
batch_size = 32
epochs = 100

def build_model():
    # constructing the model using a pre-trained model on ImageNet
    model = applications.mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3),
                                  pooling='avg')

    # only train the last 10 layers
    for layer in model.layers[:-10]:
        layer.trainable = False

    # output for all 15 classes
    predictions = Dense(15, activation="softmax")(model.output)

    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)
    
    return model_final

model_final = build_model()
# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    rotation_range=360)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,
    fill_mode="nearest",
    zoom_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    # save_to_dir = root_visualizaion_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("currency_detector.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

tensorboard = TensorBoard(log_dir='/home/xiaoyzhu/notebooks/currency_detector/tensorboard/',  write_images = True)

# Train the model
model_final.fit_generator(
    train_generator,
    steps_per_epoch = train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = validation_steps,
    workers=16,
    callbacks=[checkpoint, early, tensorboard])