#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

import rospy
import pandas as pd
import spacy
from std_srvs.srv import Empty, Trigger
from voice.srv import Stt, Tts, Vad

class SimpleRecognizer:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp_list = []
    
    def load_nlp(word_list):
        for word in word_list:
            print(word)
            self.nlp_list.append(nlp(str(word)))
        return self.nlp_list

    def __call__(self, phrase, nlp_list):
        nlp_phrase = self.nlp(phrase)
        ranks = []
        for element in nlp_list:
            ranks.append({})
            ranks[-1]['text'] = element.text
            ranks[-1]['similarity'] = nlp_phrase.similarity(element)
        return sorted(ranks, key=lambda x: x['similarity'], reverse=True)

class VoiceDemo:
    
    def __init__(self):
        rospy.init_node("voicedemo")
        print("wait_for_service('/wake_word')")
        rospy.wait_for_service('/wake_word')
        print("wait_for_service('/vad')")
        rospy.wait_for_service('/vad')
        print("wait_for_service('/tts')")
        rospy.wait_for_service('/tts')
        rospy.wait_for_service('/stt')	

        self.vad = rospy.ServiceProxy('/vad', Vad)
        self.tts = rospy.ServiceProxy('/tts', Tts)
        self.stt = rospy.ServiceProxy('/stt', Stt)
        self.wake_word = rospy.ServiceProxy('/wake_word', Empty) 
        self.recognizer = SimpleRecognizer(["What is your name?"])

    def __call__(self):
        print('Waiting wake word...')
        self.wake_word()
        print('Listening command...')
        vad_response = self.vad()
        print(vad_response)
        stt_response = self.stt(vad_response.audio_path)
        print(stt_response)
        # tts_response = self.tts(stt_response.transcription)
        result = self.recognizer(stt_response)
        print(result)
        if len(result) > 0:
            self.tts(f"I recognize the command {result[0]}")
        else:
            self.tts("I did not recognize your command")

if __name__ == "__main__":
    demo = VoiceDemo()
    while(True):
        demo()
