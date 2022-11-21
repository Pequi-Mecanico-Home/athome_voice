#!/usr/bin/env python3
import os
import time
import datetime
from abc import ABC

import torch
import rospy
from scipy.io.wavfile import write
import numpy as np

from jetson_voice.utils import audio_to_int16
from voice.msg import Audio
from voice.srv import Tts, TtsResponse
from std_msgs.msg import String

class TTSROS(ABC):
    pass

class TacotronROS(TTSROS):
    def __init__(self):  
        super().__init__()  
        
        # create topics
        self.text_subscriber = rospy.Subscriber('tts_text', String, self.text_listener, 10)
        self.audio_publisher = rospy.Publisher('tts_audio', Audio, 10)

        print("Initializing tacotron")            
        self.tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.tacotron2 = self.tacotron2.to('cuda')
        self.tacotron2.eval()

        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow = self.waveglow.to('cuda')
        self.waveglow.eval()

        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        sequences, lengths = self.utils.prepare_input_sequence(["warmup" for _ in range(5)])

        start_time = time.time() 

        with torch.no_grad():
            mel, _, _ = self.tacotron2.infer(sequences, lengths)
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        self.rate = 22050

        write(f"warmup.wav", self.rate, audio_numpy)

        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time}")
        
    def text_listener(self, msg):
        text = msg.data.strip()
        
        if len(text) == 0:
            return
            
        print(f"running TTS on '{text}'")
        
        sequences, lengths = self.utils.prepare_input_sequence([text])

        start_time = time.time() 

        with torch.no_grad():
            mel, _, _ = self.tacotron2.infer(sequences, lengths)
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        self.rate = 22050

        audio_path = f"{datetime.datetime.now()}.wav"
        write(audio_path, self.rate, audio_numpy)

        end_time = time.time()

        print(f"Elapsed time: {end_time - start_time}")
        
        samples = audio_to_int16(audio_numpy)
        
        # publish message
        msg = Audio()
        
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.model_name

        msg.info.channels = 1
        msg.info.sample_rate = self.tts.sample_rate
        msg.info.sample_format = str(samples.dtype)
        
        msg.data = samples.tobytes()
        
        self.audio_publisher.publish(msg)

        return audio_path

    def __call__(self, req):
        return self.text_listener(req)
        

if __name__ == "__main__":
    tts = TacotronROS()
    
    def handler(req):
        print(req)
        audio_path = tts(req) 
        return TtsResponse(
            audio_path=audio_path
        )
    rospy.init_node('tts')
    service = rospy.Service('voice/tts', Tts, handler)   
    
    rospy.spin()