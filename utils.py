from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color

import keras
from keras.models import model_from_json

import numpy as np
import json
from tensorforce.agents import Agent

def get_feature_extractor_model(model_path, json_path):
    json_file = open(json_path, 'r')
    intermediate_layer_model = json_file.read()
    json_file.close()

    intermediate_layer_model = model_from_json(intermediate_layer_model)
    intermediate_layer_model.load_weights(model_path)

    intermediate_layer_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return intermediate_layer_model

def get_features(feature_extractor_model, img_batch, img_size=105):
    batch_size = len(img_batch)
    img_batch = np.array(img_batch)
    img_batch = img_batch.reshape(batch_size, img_size, img_size, 1)
    intermediate_output = feature_extractor_model.predict(img_batch)
    return intermediate_output

def message_to_image(strokes, img_rows, img_cols, stroke_width=5):

    draw =  Drawing()
    draw.stroke_color = Color('white')
    draw.stroke_width = stroke_width
    draw.fill_color = Color('black')
    image =  Image(width=img_rows,
           height=img_cols,
           background=Color('black'))
    for message in strokes:
        sx, sy,cx, cy, ex, ey = message
        sx = sx*10
        sy = sy*10
        cx = cx*10
        cy = cy*10
        ex = ex*10
        ey = ey*10
        points = [(sx, sy),  # Start point
                  (cx, cy),  # First control
                  (cx, cy),  # Second control
                  (ex, ey)]  # End point
        draw.bezier(points)
        draw(image)

    img = np.array(image)
    img = img[:, :, 0]
    img = img/255
    return img

def message_to_image_sm(strokes, h=10, w=10, stroke_width=2):

    draw =  Drawing()
    draw.stroke_color = Color('white')
    draw.stroke_width = stroke_width
    draw.fill_color = Color('black')
    image =  Image(width=w,
           height=h,
           background=Color('black'))
    for message in strokes:
        sx, sy,cx, cy, ex, ey = message
        points = [(sx, sy),  # Start point
                  (cx, cy),  # First control
                  (cx, cy),  # Second control
                  (ex, ey)]  # End point
        draw.bezier(points)
        draw(image)

    img = np.array(image)
    img = img[:, :, 0]
    img = img/255
    return img

def get_agents(ppo_config, network_config, sender_type, n_features, img_dim, img_features_len, n_samples, action_size):

    with open(ppo_config) as fp:
        agent_spec_big = json.load(fp)

    with open(network_config) as fp:
        network_deep = json.load(fp)

    if sender_type=="agnostic":
        speaker_state_shape = (n_features+img_dim*img_dim,)
    elif sender_type=="aware":
        speaker_state_shape = (n_samples*n_features+img_dim*img_dim,)

    Speaker = Agent.from_spec(
        spec=agent_spec_big,
        kwargs=dict(
            states=dict(type='float', shape=speaker_state_shape),
            actions=dict(type='int', num_actions=img_dim, shape=(action_size,)),
            network=network_deep,
        )
    )

    Listener = Agent.from_spec(
        spec=agent_spec_big,
        kwargs=dict(
            states=dict(type='float', shape=(n_samples*n_features + img_features_len,)),
            actions=dict(type='int', num_actions=n_samples),
            network=network_deep,
        )
    )
    Speaker.reset()
    Listener.reset()

    return Speaker, Listener
