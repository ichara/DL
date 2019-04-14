
import os
import sys
from Bot.Config import FTG_PATH
print(FTG_PATH)
sys.path.append(os.path.join(FTG_PATH, 'python'))
from py4j.java_gateway import get_field

class SandBag(object):
    def __init__(self, gateway):
        self.gateway = gateway

    def getCharacter(self):
        return "LUD"

    def close(self):
        pass

    def getInformation(self, frameData):
        pass

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        pass
    	# print(x)
    	# print(y)
    	# print(z)
    	
    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
    	pass

    def initialize(self, gameData, player):
        return 0

    def input(self):
        pass

    def processing(self):
        return

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
