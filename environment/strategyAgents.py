from game import Agent
from game import Directions
import numpy as np


class globalAgent(Agent):
    def __init__(self, index=0, **args):
        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []

    def getAction(self, state):
        legal = state.getLegalActions(self.index)
        move = np.random.choice(legal,1)[0]
        print(move)
        if move == "West":
            move = Directions.WEST
        elif move == 'Stop':
            move = Directions.STOP
        elif move == 'East':
            move = Directions.EAST
        elif move == "Up":
            move = Directions.NORTH
        elif move == "Down":
            move = Directions.SOUTH
        return move

    def getMove(self, legal):
        move = Directions.STOP
        if (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move
