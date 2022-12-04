#!/usr/bin/env python3
import os
DIRECTORY = os.path.dirname(os.path.realpath(__file__))

import time
from datetime import datetime
from abc import ABC
from multiprocessing import Process

import rospy
import pyloudnorm as pyln
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from soundfile import SoundFile

from voice.msg import Audio
from voice.srv import Asr, AsrResponse
from std_msgs.msg import String


class ASR(ABC):
    
    def transcribe_audio_path(self, audio_path, *kwargs):
        raise NotImplementedError

    def transcribe_audio(self, audio):
        raise NotImplementedError

    def transcribe_audio(self, audio):
        raise NotImplementedError

class WhisperROS(ASR):

    def __init__(self, model_name="base", sample_rate=16000, save_dir="/logs"):
        super().__init__()    
        
        self.save_dir = save_dir
        self.sample_rate = sample_rate

        # create topics
        print(f"creating topics")
        self.audio_subscriber = rospy.Subscriber('audio_in', Audio, self.audio_listener, queue_size=10)
        self.transcript_publisher = rospy.Publisher('transcripts', String, queue_size=10)

        # load the ASR model
        print(f"loading model")        
        self.asr = whisper.load_model(model_name)
        self.decoding_options = whisper.DecodingOptions(
            language="en", without_timestamps=True, beam_size=1)
        # warmup
        print(f"running warmup")
        mel = whisper.log_mel_spectrogram(torch.empty(torch.Size([480000])).to("cuda"))
        self.asr.decode(mel, self.decoding_options)
        
        print(f"model '{model_name}' ready") 

    def transcribe(self, audio, decoding_options=None):
        if decoding_options is None:
            decoding_options = self.decoding_options

        audio = whisper.pad_or_trim(audio.flatten())
        audio = torch.from_numpy(audio).to("cuda")
        
        st = time.time()
        mel = whisper.log_mel_spectrogram(audio)
        result = self.asr.decode(mel, decoding_options)

        now_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        with open(os.path.join(self.save_dir, f'asr-{now_str}.txt'), 'w') as txt:
            print(result.text, file=txt)
        print(f"{result}")
        end = time.time()
        print(f"\ttook {end-st} seconds")
        
        msg = String()
        msg.data = result.text
        self.transcript_publisher.publish(msg)

        return result.text
        
    def audio_listener(self, msg):
        if msg.info.sample_rate != self.sample_rate:
            print(f"audio has sample_rate {msg.info.sample_rate}, "
                                      f"but ASR expects sample_rate {self.asr.sample_rate}")
            
        samples = np.frombuffer(msg.data, dtype=msg.info.sample_format)
        print(f'received audio samples {samples.shape} dtype={samples.dtype}') # rms={np.sqrt(np.mean(samples**2))}')
        
        self.transcribe(audio=samples)

    def transcribe_audio_path(self, audio_path, *kwargs):
        print(f"Starting transcribing audio {audio_path}")
        try:
            audio = whisper.load_audio(file=audio_path, sr=self.sample_rate)
            transcription = self.transcribe(audio)
        except Exception as e:
            print(f"Trascribing {audio_path}: {str(e)}")
            raise e
            return None
        return transcription

if __name__ == "__main__":
    asr = WhisperROS()
    
    def handler(req):
        print(req)
        transcription = asr.transcribe_audio_path(req.audio_path)  # TODO: refactor to use audio instead of audio path
        print(transcription)
        return AsrResponse(
            transcription=transcription
        )
    rospy.init_node('whisper_asr')
    service = rospy.Service('voice/asr', Asr, handler)   
    
    rospy.spin()

# source devel/setup.bash && roslaunch voice asr.launch &
# rosservice call voice/asr