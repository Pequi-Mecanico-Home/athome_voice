#!/usr/bin/env python3
import os
directory = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(os.path.abspath(__file__))

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

def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def interrupt_callback():
    global interrupted
    return interrupted

signal.signal(signal.SIGINT, signal_handler)

def play_audio(filename: str):
    """
    Args:
        filename: Audio filename
    """

    player = 'play' if platform.system() == 'Darwin' else 'aplay'
    Popen([player, '-q', filename])


def activate_notify():
    # audio = directory+'/resources/activate.wav'
    # audio = TOP_DIR+'/resources/okay2.wav'
	os.system(f'aplay {TOP_DIR}/resources/okay2.wav -D hw:2,0')
    #audio = abspath(dirname(abspath(__file__)) + '/../' + audio)

    # play_audio(audio)

class Precise():
	def __init__(self, model: str, callback):
		engine = PreciseEngine(directory+'/resources/precise-engine', directory+model)
		self.runner = PreciseRunner(engine, on_activation=callback)

	def start(self):
		self.runner.start()

class Snowboy():
	def __init__(self, model: str, callback):
		self.detector = snowboydecoder.HotwordDetector(directory+model, sensitivity=0.6)
		self.callback = callback

	def start(self):
		self.detector.start(detected_callback=self.callback, interrupt_check=interrupt_callback, sleep_time=0.03)

	def terminate(self):
		self.detector.terminate()

class WakeWord():
	def __init__(self):
		print('Starting... ')
		#self.engine = Precise('/resources/hey-mycroft-2', self.hotword_detected)
		self.engine = Snowboy('/resources/computer.umdl', self.hotword_detected)
		rospy.init_node('wake_word', anonymous=True)
		self.service = rospy.Service('zordon/wake_word', srv.Empty, self.activate)
		self.pub = rospy.Publisher('zordon/wake_work/detected', Empty)
		self.active = True
		self.detected = False

	def activate(self, req):
		print(req)
		self.active = True
		self.detected = False
		while(not self.detected):
			time.sleep(0.1)
		return srv.EmptyResponse()

	def run(self):
		print("starting wakeword listener...")
		self.engine.start()
		print("CTRL+C - terminating wakeword")
		self.engine.terminate()

	def hotword_detected(self):
		print("hotword detected")
		if(self.active):
			activate_notify()
			msg = Empty()
			self.pub.publish(msg)
			self.active = False
			self.detected = True
			print("hotword detected")
		self.engine.terminate()

if __name__ == "__main__":
	wakeword = WakeWord()
	wakeword.run()
	import time
