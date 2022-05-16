#!/usr/bin/env python

import asyncio
import websockets
import sys
import logging
import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import subprocess
import shutil
import pyaudio
from queue import Queue
from threading import Thread
import time
from scipy.io.wavfile import write
import random
import string
import scipy.signal as sps
import io
import soundfile as sf
import json

sys.path.append("./multilingual_kws/")
from multilingual_kws.embedding import transfer_learning, input_data

settings = None
silence_threshold = 20 / 32767
data = None    
sample_rate=16000

models = {
    "LV": tf.keras.models.load_model('models/TildePls_lv_2284shot'),
    "LT": tf.keras.models.load_model('models/TildePls_lt_2022shot'),
    "ET": tf.keras.models.load_model('models/TildePls_et_2022shot'),
    "EN": tf.keras.models.load_model('models/TildePls_en_2100shot'),
    "RU": tf.keras.models.load_model('models/TildePls_ru_2100shot')
}

def get_model():
    return tf.keras.models.load_model('models/TildePls_lv_2284shot')

def get_spectrogram(data):
    return input_data.to_micro_spectrogram(settings, data)

def detect_triggerword_spectrum(model, x):
    return model.predict(np.array([x]))

def has_new_triggerword(predictions, threshold=0.95):
    for p in predictions:
        if p[2] >= threshold:
            return True
    return False

async def handler(websocket):
    async for message in websocket:
        #logging.info(message)
        obj = json.loads(message)        
        audio_bytes = bytes(obj["audio"])
        #audio, message_sample_rate = tf.audio.decode_wav(message, desired_channels=1, desired_samples=44100)
        audio, message_sample_rate = sf.read(io.BytesIO(audio_bytes))
        if np.abs(audio).mean() < silence_threshold:
            logging.info("Signal too quiet")
            await websocket.send('0')
        else:
            if message_sample_rate != sample_rate:
                samples = round(len(audio) * float(sample_rate) / message_sample_rate)
                audio = sps.resample(audio, samples)
            #fname = f"output/audio_{''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))}.wav"
            #write(fname, 16000, np.array(audio).astype(np.float32))
            
            spectrogram = get_spectrogram(audio)
            predictions = detect_triggerword_spectrum(models[obj["model"]], spectrogram)
            wakeword_detected_pred = predictions[0][2]
            await websocket.send(str(wakeword_detected_pred))


async def main():
    global settings
    settings = input_data.standard_microspeech_model_settings(label_count=1)
    logging.info("Started")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
asyncio.run(main())