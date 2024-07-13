from tensorflow import keras
from catvsdog_model.config.core import config

# Performing the data augmentation as series of transformations
def get_data_augmented(flip, rotation, zoom):
    data_augmentation = keras.Sequential([keras.layers.RandomFlip(flip),
                                          keras.layers.RandomRotation(rotation),
                                          keras.layers.RandomZoom(zoom)])

    return data_augmentation


data_augmentation = get_data_augmented(flip = config.model_configs.flip, 
                                       rotation = config.model_configs.rotation, 
                                       zoom = config.model_configs.zoom)
