#!/usr/local/bin/python3.6
import os
directory = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
import time
import numpy as np
# from precise_runner import PreciseEngine
from precise import PreciseRunner

import rospy
from std_msgs.msg import Empty
import std_srvs.srv as srv

import time
import sys
import signal

import snowboydecoder

import platform
from subprocess import Popen

interrupted = False
from soundfile import SoundFile
import soundfile as sf

import rospy

from std_srvs.srv import Empty, EmptyResponse

import logging

def play_audio(filename: str):
    player = 'play' if platform.system() == 'Darwin' else 'aplay'
    Popen([player, '-q', filename])

def activate_notify():
    os.system(f'aplay {TOP_DIR}/resources/okay2.wav -D hw:2,0')

interrupted = False

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def interrupt_callback():
    global interrupted
    return interrupted

class WakeWord:

    def __init__(self, 
                 input_device=24,
                 model="/voice_ws/src/voice/src/resources/computer.umdl"):
                 
        print(f"Creating wakeword")
        self.service = rospy.Service('wake_word', Empty, self) 
        
        self.detector = snowboydecoder.HotwordDetector(model, sensitivity=0.8)

        activate_notify()

    def callback(self):
        self.detector.terminate()

    def __call__(self, req):
        print(req)
        self.detector.start(detected_callback=self.callback, interrupt_check=interrupt_callback, sleep_time=0.03)
        print("Hotword detected")
        return EmptyResponse()


if __name__ == "__main__":
    rospy.init_node('wake_word')
    wakeword = WakeWord()
    rospy.spin()
