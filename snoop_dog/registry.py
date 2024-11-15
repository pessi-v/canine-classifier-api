from tensorflow import keras
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import io
import random
import string

def load_and_resize_image(img):
    img = Image.open(img).convert("RGB")
    print("image loaded")
    img = img.resize((256,256))
    print("image resized")
    return img

def dogimg_process(img):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_dogmodel():
    new_model = keras.models.load_model(f"{os.path.dirname(os.path.dirname(__file__))}/snoop_dog/models/effnetv2s_finetuned_valsplit01.keras")
    return new_model

def gradcam(model, img_array):
    last_conv_layer = model.get_layer('efficientnetv2-s').get_layer('top_activation')
    last_conv_layer_model = tf.keras.Model(model.get_layer('efficientnetv2-s').inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["global_average_pooling2d", "dense"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        inputs = img_array
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (256, 256))
    fig = plt.figure(visible=True, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.] )
    fig.set_size_inches(1,1)
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(img_array[0, :, :, :] / 255., aspect='auto')
    plt.imshow(gradcam, alpha=0.5, aspect='auto')
    random_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    filename = './static/'+random_name+'.jpg'
    plt.savefig(filename, dpi=256)
    return filename[1:]