# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]



        "*** YOUR CODE HERE ***"
        ghostPos = newGhostStates[0].configuration.getPosition()
        foodValue = 0
        ghostValue = 0
        scaredVal = 0
        min = 10 ** 5
        for i in range(0, len(newFood.asList())):
            dist = abs(newPos[0] - newFood.asList()[i][0]) + abs(newPos[1] - newFood.asList()[i][1])
            if (dist < min):
                min = dist
        foodValue = 1 / min



        ghostDist = abs(newPos[0] - ghostPos[0]) + abs(newPos[1] - ghostPos[1])
        if (ghostDist == 0):
            ghostValue = -foodValue
        if (newScaredTimes[0] != 0):
            ghostValue = 1 / ghostDist
        final = successorGameState.getScore() + foodValue + ghostValue





        return final

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value = self.max_value_state(gameState, 0, 0)
        return value[1]

    def max_value_state(self, gameState: GameState, agentIndex, depth):
        actionsList = gameState.getLegalActions(0)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isWin()):
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isLose()):
            return (self.evaluationFunction(gameState), 0)
        if (depth == self.depth):
            return (self.evaluationFunction(gameState), 0)

        value = float('-inf')
        move = 0
        newIndex = agentIndex + 1
        for i in range(0, len(actionsList)):
            node = self.min_value_state(gameState.generateSuccessor(0, actionsList[i]), newIndex, depth)
            num = node[0]
            if (num > value):
                value = num
                move = actionsList[i]
        return (value, move)

    def min_value_state(self, gameState: GameState, agentIndex, depth):
        actionsList = gameState.getLegalActions(agentIndex)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        value = float('inf')
        move = 0
        for i in range(0, len(actionsList)):
            if (agentIndex == gameState.getNumAgents() - 1):
                node = self.max_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), 0, depth + 1)
            else:
                node = self.min_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), agentIndex + 1, depth)
            num = node[0]
            if (num < value):
                value = num
                move = actionsList[i]
        return (value, move)






















class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = self.max_value_state(gameState, 0, 0, float('-inf'), float('inf'))
        return value[1]
    def max_value_state(self, gameState: GameState, agentIndex,depth, alpha, beta):
        actionsList = gameState.getLegalActions(0)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isWin()):
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isLose()):
            return (self.evaluationFunction(gameState), 0)
        if (depth == self.depth):
            return (self.evaluationFunction(gameState), 0)
        newIndex = agentIndex + 1
        value = float('-inf')
        move = 0
        for i in range(0, len(actionsList)):
            node = self.min_value_state(gameState.generateSuccessor(0, actionsList[i]), newIndex, depth, alpha, beta)

            num = node[0]
            if (num > value):
                value = num
                move = actionsList[i]
            if (value > beta):
                return (value, move)
            alpha = max(alpha, value)
        return (value, move)
    def min_value_state(self, gameState: GameState, agentIndex, depth, alpha, beta):
        actionsList = gameState.getLegalActions(agentIndex)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        value = float('inf')
        move = 0
        for i in range(0, len(actionsList)):
            if (agentIndex == gameState.getNumAgents() - 1):
                node = self.max_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), 0, depth + 1, alpha, beta)
            else:
                node = self.min_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), agentIndex + 1, depth, alpha, beta)
            num = node[0]
            if (num < value):
                value = num
                move = actionsList[i]
            if (value < alpha):
                return (value, move)
            beta = min(beta, value)
        return (value, move)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = self.max_value_state(gameState, 0, 0)
        return value[1]

    def max_value_state(self, gameState: GameState, agentIndex, depth):
        actionsList = gameState.getLegalActions(0)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isWin()):
            return (self.evaluationFunction(gameState), 0)
        if (gameState.isLose()):
            return (self.evaluationFunction(gameState), 0)
        if (depth == self.depth):
            return (self.evaluationFunction(gameState), 0)
        newIndex = agentIndex + 1
        value = float('-inf')
        move = 0
        for i in range(0, len(actionsList)):
            node = self.exp_value_state(gameState.generateSuccessor(0, actionsList[i]), newIndex, depth)

            num = node[0]
            if (num > value):
                value = num
                move = actionsList[i]

        return (value, move)

    def exp_value_state(self, gameState: GameState, agentIndex, depth):
        actionsList = gameState.getLegalActions(agentIndex)
        if len(actionsList) == 0:
            return (self.evaluationFunction(gameState), 0)
        value = 0
        move = 0
        for i in range(0, len(actionsList)):
            if (agentIndex == gameState.getNumAgents() - 1):
                node = self.max_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), 0, depth + 1)
            else:
                node = self.exp_value_state(gameState.generateSuccessor(agentIndex, actionsList[i]), agentIndex + 1,
                                            depth)
            num = node[0]
            p = 1 / len(actionsList)
            value += p * num
            move = actionsList[i]


        return (value, move)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    min = float('inf')
    dist = 0
    for i in range(0, len(newFood.asList())):
        dist = abs(newPos[0] - newFood.asList()[i][0]) + abs(newPos[1] - newFood.asList()[i][1])
        if (dist < min):
            min = dist
    foodValue = 1 / min

    min = float('inf')
    closeGhosts = 0
    ghostValue = 0
    foodWeight = 1
    ghostWeight = 1
    min = float('inf')
    for i in range(0, len(newGhostStates)):
        ghostDist = abs(newPos[0] - newGhostStates[i].configuration.getPosition()[0]) + abs(newPos[1] - newGhostStates[i].configuration.getPosition()[1])
        if (ghostDist < min):
            min = ghostDist

    if (min <= 1):
        closeGhosts = -foodValue
    if (newScaredTimes[0] != 0):
        foodWeight += 4
        ghostWeight = 0




    numCapsule = len(currentGameState.getCapsules())
    final = currentGameState.getScore() + (foodWeight * foodValue) + (ghostWeight*closeGhosts) - numCapsule
    return final







# Abbreviation
better = betterEvaluationFunction
