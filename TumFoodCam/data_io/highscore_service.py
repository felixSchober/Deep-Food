import os
import pickle
import operator
from misc import utils

class HighscoreService(object):
    """Deprecated High score class."""

    HIGHSCORES_STANDARD_PATH = utils.get_data_path() + "Highscores.dat"

    def __init__(self):
        self.highscores = []


    def insert_score(self, score, description):
        """ Inserts a score and a description and resorts the resulting list of scores."""
        self.highscores.append((score, description))
        self.__sort_scores()

        # calculate position
        position = 0
        for i in range(len(self.highscores)):
            s, _ = self.highscores[i]
            if s == score:
                position = i + 1

        self.save()
        return position

    def get_position(self, position):
        """ Returns the score for a position."""
        return self.highscores[position - 1]
    
    def print_surrounding_highscores(self, position):
        """ 
        Prints the surrounding scores around a position.
        Takes two scores before and two after position.
        """

        # prevent overflow. Positions are naturally measured from 1 to n (not from 0 to n-1)
        betterPosition = max(1, position - 2)
        worsePosition = min(len(self.highscores), position + 2)
        higherScores = self.highscores[betterPosition-1:position-1]
        lowerScores = self.highscores[position:worsePosition]

        own_score, own_description = self.highscores[position-1]

        for score, description in higherScores:
            print "\t\t[{0}]\t{1}".format(str(round(score,3)), description)
        print "**\t{0}.\t[{1}]\t{2} **".format(position, str(round(own_score,3)), own_description)
        for score, description in lowerScores:
            print "\t\t[{0}]\t{1}".format(str(round(score,3)), description)
         
    def print_highscore(self):
        """ Prints the complete high score list to the console."""
        for i in range(len(self.highscores)):
            score, description = self.highscores[i]
            print "\t{0}.\t[{1}]\t{2}".format(i + 1, str(round(score,3)), description)

    def __sort_scores(self):
        self.highscores = sorted(self.highscores, key=operator.itemgetter(0), reverse=True)

    def load(self):        
        if not os.path.isfile(HighscoreService.HIGHSCORES_STANDARD_PATH):
            self.highscores = []
            return self

        with open(HighscoreService.HIGHSCORES_STANDARD_PATH, "r") as f:
            self = pickle.load(f)

        return self

    def save(self):        
        with open(HighscoreService.HIGHSCORES_STANDARD_PATH, "wb") as f:
            pickle.dump(self,f)
            




