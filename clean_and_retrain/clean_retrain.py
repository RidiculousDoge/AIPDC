"""
Main routine for training the poisoned net.
"""
# pylint: disable-msg=C0103
# Above line turns off pylint complaining that "constants" aren't ALL_CAPS
import argparse
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import eval_badnet
import gtsrb_dataset
from keras.applications.vgg16 import VGG16

def build_model(num_classes=43):
    """
    Build the 6 Conv + 2 MaxPooling NN. Paper did not specify # filters so I
    picked some relatively large ones to start off.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def train(dataset,origin_model,epochs=10):
    """
    Train a model on the GTSRB dataset
    """
    conv_model = build_model()
    conv_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    infos=origin_model.split('-')
    poison_type=infos[1]
    poison_loc=infos[2]
    poison_size=int(infos[3])
    random=infos[4]
    if(len(random)>10):
        random=random[13:]
        random=(random=='True')
    else:
        random=False
    filepath = "output/retrain-"+"{}".format("random" if random else '')\
        +"-{}".format(poison_type if poison_type else 'clean') \
        + '-%s-%d-{epoch:02d}-{val_acc:.2f}.hdf5'%(poison_loc,poison_size)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]


    history = conv_model.fit(dataset.train_images, dataset.train_labels,
                             callbacks=callbacks_list, epochs=epochs,
                             validation_data=(dataset.test_images,
                                              dataset.test_labels))

    test_loss, test_acc = conv_model.evaluate(dataset.test_images,
                                              dataset.test_labels, verbose=2)
    print("Test Loss: {}\nTest Acc: {}".format(test_loss, test_acc))
    eval_badnet.evaluate_model(conv_model=conv_model)


