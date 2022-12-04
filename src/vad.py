#!/usr/bin/env python3
import os
DIRECTORY = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from datetime import datetime
from multiprocessing import Process

import time
import numpy as np
import pyloudnorm as pyln
import rospy
import soundfile as sf
import torch
import torchaudio
import whisper
from soundfile import SoundFile

from jetson_voice import AudioInput
from jetson_voice.utils import audio_to_float, audio_to_int16
from voice.msg import Audio
from std_msgs.msg import Empty
from voice.srv import Vad, VadResponse


class VADRecorder:

    def __init__(self, 
                 max_samples=16_000*6,
                 sample_rate=16_000, 
                 normalize=True,
                 mic=0,  # ToDo: pass as argument
                 target_norm_lufs=-12.0,
                 warmup=5,
                 audio_prefix='',
                 save_dir="/logs",
                 chunk_size=4_000,
                 background_detection_patience=(16_000*4)//4_000):    
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.mic = mic
        self.prefix = audio_prefix
        self.background_detection_patience = background_detection_patience
        self.normalize = normalize
        self.max_samples = max_samples
        self.background_detection_patience = background_detection_patience
        self.target_norm_lufs = target_norm_lufs
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # create model
        print(f"Creating VAD model")
        torch.__version__ = torch.__version__.split('a')[0]  # Workaround to avoid errors during silero load
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',  # TODO: load from local directory
                                                model='silero_vad',
                                                force_reload=True,
                                                onnx=False)
        self.model.to("cuda")
        get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.utils
        self.vad_iterator = VADIterator(self.model)
        print(f"Creating VAD model: DONE")
                
        print(f"Testing mic streaming on device {self.mic}")
        try:
            stream = AudioInput(mic=self.mic, 
                                sample_rate=self.sample_rate, 
                                chunk_size=self.chunk_size)
        except Exception as e:
            print(f"ERROR creating streaming on device {self.mic}")
        for i in range(5):
            prob = self.model(torch.rand((1, 16_000), dtype=torch.float32).to("cuda"), 16_000).item()
            print(prob)

        # create topics
        print(f"creating topics")
        self.audio_publisher = rospy.Publisher('audio_in', Audio, queue_size=10)
        self.record_subscriber = rospy.Subscriber('vad_start', Empty, queue_size=10)

    def int2float(sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def audio_to_int16(samples):
        if samples.dtype == np.int16:
            return samples
        elif samples.dtype == np.float32:
            return (samples * 32768).astype(np.int16)
        else:
            return samples.astype(np.int16)

    def publish_audio(self, samples):
        if samples.dtype == np.float32:  # convert to int16 to make the message smaller
            samples = audio_to_int16(samples)

        if samples.dtype != np.int16:  # the other voice nodes expect int16/float32
            raise ValueError(f'audio samples are expected to have datatype int16, but they were {samples.dtype}')
        
        print(f'publishing audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        # publish message
        msg = Audio()

        msg.header.stamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        msg.header.frame_id = f"Mic/device ID: {self.mic}"

        msg.info.channels = 1  # AudioInput is set to mono
        msg.info.sample_rate = self.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)
        print("publish audio: Ok")

    def vad_record(self, save_audio=True):
        prob = self.model(torch.rand((1,16_000), dtype=torch.float32).to("cuda"), 16_000).item()
        os.system(f'aplay resources/okay2.wav -D hw:2,0')  # TODO: refactor path or use sound_play
        
        print(f"Starting mic streaming on device {self.mic}")
        try:
            stream = AudioInput(mic=self.mic, 
                                sample_rate=self.sample_rate, 
                                chunk_size=self.chunk_size)
        except Exception as e:
            print(f"ERROR creating streaming on device {self.mic}")
            return None

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        background_detection_patience = self.background_detection_patience

        final_samples = []
        all_samples = []

        voice_detected = False
        while background_detection_patience > 0:
            samples = next(stream)
            if len(samples) < self.chunk_size:
                break
            prob = self.model(torch.tensor(VADRecorder.int2float(samples)).to("cuda"), 16_000).item()
            print(prob)

            if prob > 0.9:
                background_detection_patience = self.background_detection_patience
                print("Detected voice")
                voice_detected = True
            else:
                background_detection_patience -= 1
                print(background_detection_patience)

            if voice_detected:
                final_samples = [*final_samples, *samples]
            all_samples = [*all_samples, *samples]

            if len(final_samples) > self.max_samples: 
                break
            
        self.vad_iterator.reset_states() 

        if len(final_samples) == 0:
            final_samples = all_samples
        
        print('Closing stream')
        stream.close()
        print('\naudio stream closed.')

        audio_name = self.prefix+current_time+".wav"
        audio_path = os.path.join(self.save_dir, audio_name)
        output_wav = SoundFile(audio_path, mode='w', samplerate=16_000, channels=1)
        output_wav.write(final_samples)

        audio = final_samples = np.array(final_samples)
        
        print('Audio shape:', final_samples.shape)

        if self.normalize:
            data, rate = sf.read(audio_path) # load audio

            audio_name = self.prefix+current_time+".norm.wav"
            audio_path = os.path.join(self.save_dir, audio_name)
            # peak normalize audio to -1 dB
            peak_normalized_audio = pyln.normalize.peak(data, -1.0)

            # measure the loudness first 
            meter = pyln.Meter(rate) # create BS.1770 meter
            loudness = meter.integrated_loudness(data)

            # loudness normalize audio to -12 dB LUFS
            audio = loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.target_norm_lufs)

            output_wav_norm = SoundFile(audio_path, mode='w', samplerate=16_000, channels=1)
            output_wav_norm.write(audio)

        os.system(f'aplay resources/activate.wav -D hw:2,0')  #TODO: use ros library
                    
        print(f"Saving audio {os.path.abspath(audio_path)}")
        
        self.publish_audio(VADRecorder.audio_to_int16(audio))
        
        return audio_path

    def __call__(self, save_audio=True):
        return self.vad_record(save_audio=save_audio)

class VADRecorderService(VADRecorder):

    def __init__(self, name='voice/vad', **kwargs):
        super().__init__(**kwargs)
        # create service
        print(f"creating service {name}")
        self.service = rospy.Service(name, Vad, self)

    def __call__(self, req=None):
        audio_name = self.vad_record(save_audio=True)
        return VadResponse(
            audio_path=os.path.abspath(audio_name)
        )

if __name__ == "__main__":
    rospy.init_node('vad')
    vad_recorder_service = VADRecorderService()
    print('ready')
    rospy.spin()

# source devel/setup.bash && roslaunch voice vad.launch &
# rosservice call voice/vad