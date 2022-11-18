#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import requests
import json
import pandas as pd
import spacy
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)
directory = os.path.dirname(os.path.realpath(__file__))

import sys
current_module = sys.modules[__name__]

import rospy
from std_srvs.srv import Empty, Trigger
from voice.srv import stt as stt_srv
from voice.srv import tts as tts_srv
from voice.srv import vad as vad_srv

from interacao_nlu.srv import Nlu

###########################################################
# List of Intens
# AWALYS CHANGE THIS LIST WHEN ,ADD A NEW INTENT, OR CHANGE AN INTENT NAME
##########################################################
from enum import Enum
class Intents(Enum):
    MOVE = 'move'
    GREET = 'greet'
    QUESTION = 'question'
    FOLLOW = 'follow'
    AFFIRM = "affirm"
    DENY = "deny"
    START = "start"
    STOP = "stop"
    PICK = "pick"
    BYE = "bye"
    NONE = None
    

###########################################################
# Initilize some global variables
# the list of questions and answer and its nlp abstraction
##########################################################
#Questions and Answer dataset
QandA = pd.read_csv(directory+'/resources/questions_and_answers.csv', sep='|')
nlp = spacy.load('en_core_web_sm')
def load_nlp(word_list):
    global nlp
    nlp_list = []
    for word in word_list:
        print(word)
        nlp_list.append(nlp(str(word)))
    return nlp_list

def compareToNlpList(phrase, nlp_list):
    global nlp
    nlp_phrase = nlp(phrase)
    ranks = []
    for element in nlp_list:
        ranks.append({})
        ranks[-1]['text'] = element.text
        ranks[-1]['similarity'] = nlp_phrase.similarity(element)
    return sorted(ranks, key=lambda x: x['similarity'], reverse=True)
nlp_list = load_nlp(QandA['QUESTION'])
#The database


class Planner():
    def __init__(self):
        # define the name of the action which can then be included in training stories
        x = "sup"

    def read(self, nlu_response):
        global nlp_list, QandA
        print(nlu_response)
        message = nlu_response.message
        if nlu_response.intent is None:
            intent = Intents.QUESTION
        else:
            intent = Intents(nlu_response.intent)
        entities = Entities(nlu_response.entities)
        if(intent == Intents.QUESTION):
            rank = compareToNlpList(message, nlp_list)
            if(float(rank[0]['similarity']) > 0.65):
                #Grab answer form the Q and A dataframe
                answer = QandA[QandA['QUESTION'] == rank[0]['text']]['ANSWER'].iloc[0]
                #If the answer is in the format ${code} grab the code inside and run it
                if '$' in answer:
                    code = re.search('\${(.*)}', answer).group(1)
                    answer = eval(code)
                return answer
        if(intent == Intents.GREET):
            return "Hello there, how can I help you?"
        if(intent == Intents.MOVE):
            if(entities.size() > 0):
                if(entities.hasAllOfTypes(["location", "person"])):
                    return "Ok, I will move to the " + entities.getFromTypes(["location", "direction"])[0]
    
            return "Where do you want me move?"
        if(intent == Intents.FOLLOW):
            msg = "I will start folowing"
            target = "you"
            location = ""
            if(entities.size() > 0):
                if(entities.hasType("pronoun")):
                    pronoun = entities.getFromType("pronoun")[0]
                    if(pronoun == "him" or pronoun == "her"):
                        target = pronoun
                if(entities.hasType("gender")):
                    gender = entities.getFromType("gender")[0]
                    if(gender == "male"):
                        target = "him"
                    else:
                        target = "her"
                if(entities.hasType("name")):
                    target = entities.getFromType(["name"])[0]
                if(entities.hasType("location")):
                    location = "to the " + entities.getFromType("location")[0]
            
            msg += " "+target
            msg += " "+location

            return msg
        return "Sorry I did not understand your question"

class Entities():
    def __init__(self, entities):
        self.entities = [eval(x) for x in entities]

    def hasType(self, type):
        for entity in self.entities:
            if(entity["entity"] == type):
                return True
        return False

    def hasAnyOfTypes(self, types):
        for type in types:
            if(self.hasType(type)):
                return True
        return False

    def hasAllOfTypes(self, types):
        for type in types:
            if(not self.hasType(type)):
                return False
        return True

    def countOfType(self, type):
        count = 0
        for entity in self.entities:
            if(entity["entity"] == type):
                count += 1
        return count

    def getFromType(self, type):
        typeList = []
        for entity in self.entities:
            if(entity["entity"] == type):
                typeList.append(entity["value"])
        return typeList

    def getFromTypes(self, types):
        typesList = []
        for type in types:
            typesList.extend(self.getFromType(type))
        return typesList

    def size(self):
        return len(self.entities)

    def __repr__(self):
        return str(self.entities)

class ChatBot():
    def __init__(self):
        rospy.init_node("planner")
        print("wait_for_service('/zordon/wake_word')")
        rospy.wait_for_service('/zordon/wake_word')
        print("wait_for_service('/zordon/vad')")
        rospy.wait_for_service('/zordon/vad')
        print("wait_for_service('/zordon/tts')")
        rospy.wait_for_service('/zordon/tts')
        print("wait_for_service('/zordon/whisper')")
        rospy.wait_for_service('/zordon/stt/whisper')
        # rospy.wait_for_service('/zordon/stt/w2v')
        print("wait_for_service('/zordon/nlu')")
        rospy.wait_for_service('/zordon/nlu')	

        print("========================================")

        self.vad = rospy.ServiceProxy('/zordon/vad', vad_srv)
        self.tts = rospy.ServiceProxy('/zordon/tts', tts_srv)
        self.stt = rospy.ServiceProxy('/zordon/stt/whisper', stt_srv)
        self.wake_word = rospy.ServiceProxy('/zordon/wake_word', Empty)
        self.nlu = rospy.ServiceProxy('/zordon/nlu', Nlu)
        self.planner = Planner()

    def listen(self):
        print('Waiting wake word...')
        self.wake_word()
        print('Listening command...')
        vad_response = self.vad()
        print(vad_response)
        stt_response = self.stt(vad_response.audio_path)
        print(stt_response)
        #tts_response = self.tts(stt_response.transcription)
        nlu_response = self.nlu(stt_response.transcription)
        print(nlu_response)
        self.tts(self.planner.read(nlu_response))

if __name__ == "__main__":
    chatbot = ChatBot()
    while(True):
        chatbot.listen()
