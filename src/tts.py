#!/usr/bin/env python3
import os
DIRECTORY = os.path.dirname(os.path.realpath(__file__))

import time
from datetime import datetime
from abc import ABC

import torch
import rospy
from scipy.io.wavfile import write as save_wav
import numpy as np

from jetson_voice.utils import audio_to_int16
from audio import AudioOutput  # Should be from jetson_voice import ... but is leading to an error
from voice.msg import Audio
from voice.srv import Tts, TtsResponse
from std_msgs.msg import String

class TTSROS(ABC):
    pass

class TacotronROS(TTSROS):
    def __init__(self,
                 output_device=0, # TODO: pass as argument
                 output_rate=44100,
                 warmup=5,
                 audio_prefix='',
                 cache_audios=True,
                 preload_file=os.path.join(DIRECTORY, "pre_generate_tts.txt"),
                 save_dir="/logs"):  
        super().__init__()  

        self.tts_rate = 22050
        self.output_device = output_device
        self.audio_prefix = audio_prefix
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.preloaded_audios = {}
        self.cache_audios = cache_audios
        
        # create topics
        self.text_subscriber = rospy.Subscriber('tts_text', String, self.text_listener, queue_size=10)
        self.audio_publisher = rospy.Publisher('tts_audio', Audio, queue_size=10)

        print("Initializing tacotron")            
        self.tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.tacotron2 = self.tacotron2.to('cuda')
        self.tacotron2.eval()

        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow = self.waveglow.to('cuda')
        self.waveglow.eval()

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

        start_time = time.time() 

        # Test:
        try:
            audio_device = AudioOutput(self.output_device, self.tts_rate) 
        except Exception as e:
            print(f"[ERROR] creating audio device with id={self.output_device} and rate={self.tts_rate}: {str(e)}")

        with torch.no_grad():
            if preload_file is not None:
                if warmup is not None:
                    print(f"Will not warmup the model for {warmup} times. Will use preloaded file {preload_file}.")
                with open(preload_file) as inputs:
                    for text in inputs.readlines():
                        print(f"Generating {text.strip()}")
                        # generate audio
                        sequences, lengths = self.utils.prepare_input_sequence([text])
                        mel, _, _ = self.tacotron2.infer(sequences, lengths)
                        audio = self.waveglow.infer(mel)
                        # save audio
                        audio_numpy = audio[0].data.cpu().numpy()
                        input_text_save = text.strip().replace(' ', '_')
                        now_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
                        audio_path = os.path.join(self.save_dir, f"{self.audio_prefix}{now_str}-{input_text_save}.wav")
                        save_wav(audio_path, self.tts_rate, audio_numpy)
                        # cache audio
                        self.preloaded_audios[text] = audio, audio_path
            else:
                texts = ["warmup" for _ in range(warmup)]
                for text in texts:
                    print(f"Generating {text}")
                    sequences, lengths = self.utils.prepare_input_sequence([text])
                    mel, _, _ = self.tacotron2.infer(sequences, lengths)
                    audio = self.waveglow.infer(mel)
                    save_wav(os.path.join(self.save_dir, f"{self.audio_prefix}{text}.wav"), self.tts_rate, audio_numpy)

        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time}")

    def generate(self, text, auto_play=True):        
        if len(text) == 0:
            print("Msg is empty!")
            return
            
        print(f"Generating '{text}'")
        if text in self.preloaded_audios:
            audio, audio_path = self.preloaded_audios[text]
        else:
            sequences, lengths = self.utils.prepare_input_sequence([text])
            start_time = time.time() 

            with torch.no_grad():
                mel, _, _ = self.tacotron2.infer(sequences, lengths)
                audio = self.waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()

            # Ideally, another node should be responsible to play the audio
            if auto_play:
                try:
                    audio_device = AudioOutput(self.output_device, self.tts_rate) 
                    audio_device.write(audio_numpy)
                except Exception as e:
                    print(f"[ERROR] creating audio device with id={self.output_device} and rate={self.tts_rate}: {str(e)}")

            now_str = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
            input_text_save = text.strip().replace(' ', '_')
            audio_path = os.path.join(self.save_dir, f"{self.audio_prefix}{now_str}-{input_text_save}.wav")
            save_wav(audio_path, self.tts_rate, audio_numpy)
            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time}")

            if self.cache_audios:
                self.preloaded_audios[text] = audio, audio_path
        
        return audio_path
        
    def text_listener(self, msg):
        text = msg.data.strip()
        audio_numpy = self.generate(msg, auto_play=False)
        samples = audio_to_int16(audio_numpy)
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
        msg.header.frame_id =  f"Model: Tacotron 2 | Device ID: {self.output_device}"

        msg.info.channels = 1
        msg.info.sample_rate = self.tts_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        self.audio_publisher.publish(msg)

    def __call__(self, req, auto_play=True):
        return self.generate(req, auto_play=auto_play)
        

if __name__ == "__main__":
    tts = TacotronROS()
    
    def handler(req):
        print(req)
        audio_path = tts(req.text, auto_play=True) 
        return TtsResponse(
            audio_path=audio_path
        )
    rospy.init_node('tts')
    service = rospy.Service('voice/tts', Tts, handler)   
    
    rospy.spin()

# source devel/setup.bash && roslaunch voice tts.launch &
# rosservice call voice/tts