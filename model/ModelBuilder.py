import tensorflow as tf

from model.ResNet34 import ResNet34


def build_small(classes: int, input_shape: () = (32, 32, 3)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                     input_shape=input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation('softmax'))
    return model


def build_resnet50(classes: int, input_shape: () = (224, 224, 3)) -> tf.keras.models.Model:
    base_model = tf.keras.applications.ResNet50(include_top=False,
                                                input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.keras.layers.Dense(classes, name='scores', activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=logits)
    return model


def build_resnet34(classes: int, input_shape: () = (224, 224, 3)) -> tf.keras.models.Model:
    model = ResNet34(classes, input_shape).build()
    return model
