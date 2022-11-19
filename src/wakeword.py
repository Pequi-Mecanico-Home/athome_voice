#!/usr/bin/env python3
import os
directory = os.path.dirname(os.path.realpath(__file__))

# from precise_runner import PreciseEngine
from precise import PreciseRunner

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
	os.system(f'aplay {filename} -D hw:2,0')
	# player = 'play' if platform.system() == 'Darwin' else 'aplay'
	# Popen([player, '-q', filename, '-D hw:2,0'])


def activate_notify():
    audio = directory+'/resources/okay2.wav'
    #audio = abspath(dirname(abspath(__file__)) + '/../' + audio)

    play_audio(audio)

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
		self.active = True
		self.detected = False

	def activate(self, req):
		print(req)
		self.active = True
		self.detected = False
		while(not self.detected):
			time.sleep(0.1)

	def run(self):
		print("starting wakeword listener...")
		self.engine.start()
		print("CTRL+C - terminating wakeword")
		self.engine.terminate()

	def hotword_detected(self):
		activate_notify()
		print("hotword detected")
		if(self.active):
			self.active = False
			self.detected = True
			print("hotword detected")

if __name__ == "__main__":
	wakeword = WakeWord()
	wakeword.run()
	import time
