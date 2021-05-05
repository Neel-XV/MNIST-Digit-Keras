# from keras import models
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, model_from_json


def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # loaded_model = models.load_model("my_model.h5")
    print("Loaded Model from disk")
    # compile and evaluate loaded model
    # loaded_model.compile(loss='categorical_crossentropy',
    #                      optimizer='adam', metrics=['accuracy'])
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    # graph = tf.get_default_graph()
    graph = tf.compat.v1.get_default_graph()
    return loaded_model, graph
