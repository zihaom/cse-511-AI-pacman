# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
from searchAgents import mazeDistance

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def evaluationFunction(self, currentGameState, action):
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

    ghostDistances = []
    for index in range(1, len(newGhostStates)+1):
        ghostPosition = successorGameState.getGhostPosition(index)
        ghostDistances.append(mazeDistance(newPos, ghostPosition, successorGameState))
    if min(ghostDistances) <= 2:
        return -min(ghostDistances)
    scared = []
    for scare in newScaredTimes:
        if scare > 0:
            scared.append(scare)
    if len(scared) > 0:
        deliciousGhost = []
        for ghostState in newGhostStates:
          if ghostState.scaredTimer>0:
            deliciousGhost.append(mazeDistance(newPos, ghostState.getPosition(), successorGameState))
        
        return successorGameState.getScore() +200/min(deliciousGhost)            
    if successorGameState.getScore() > currentGameState.getScore():
        return successorGameState.getScore() 
    foodDistance = []
    for food in newFood.asList():

        dis = mazeDistance(newPos, food, successorGameState)
        foodDistance.append(dis)
        if dis <=2:
          break
    if len(foodDistance) == 0:
      return successorGameState.getScore()
    return successorGameState.getScore()- min(foodDistance)

def scoreEvaluationFunction(currentGameState):
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

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    maxScore = float('-inf')
    legalActions = gameState.getLegalActions(0)
    legalActions.remove('Stop')
    a = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.minRound(successor, 1, 0)
      if (score > maxScore):
        maxScore = score
        a = action
    return a

  def minRound(self, gameState, agentID, depth):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID >= gameState.getNumAgents():
      return self.maxRound(gameState, 0, depth+1)

    legalActions = gameState.getLegalActions(agentID)
    minScore = float('inf')
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      minScore = min(minScore, self.maxRound(newGameState, agentID+1, depth))
    return minScore

  def maxRound(self, gameState, agentID, depth):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)
#only pecman gets the max round, just easier to check here than in minround
    if agentID != 0:
      return self.minRound(gameState, agentID, depth)
      
    legalActions = gameState.getLegalActions(0)
    maxScore = float('-inf')
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      maxScore = max(maxScore, self.minRound(newGameState, 1, depth))
    return maxScore

  def done(self, gameState, depth):#base case when min rounds actually gets their score
    if (gameState.isWin() or gameState.isLose() or depth == self.depth):
      return True

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    maxScore = float('-inf')
    legalActions = gameState.getLegalActions(0)
    legalActions.remove('Stop')
    a = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.minRound(successor, 1, 0, maxScore, float('inf'))
      if (score > maxScore):
        maxScore = score
        a = action
    return a

  def minRound(self, gameState, agentID, depth, alpha, beta):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID >= gameState.getNumAgents():
      return self.maxRound(gameState, 0, depth+1, alpha, beta)

    legalActions = gameState.getLegalActions(agentID)
    minScore = float('inf')
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      minScore = min(minScore, self.maxRound(newGameState, agentID+1, depth, alpha, beta))
      beta = min(beta, minScore)
      if (alpha >= beta):
        return minScore
    return minScore

  def maxRound(self, gameState, agentID, depth, alpha, beta):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID != 0:
      return self.minRound(gameState, agentID, depth, alpha, beta)
      
    legalActions = gameState.getLegalActions(0)
    maxScore = float('-inf')
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      maxScore = max(maxScore, self.minRound(newGameState, 1, depth, alpha, beta))
      alpha = max(alpha, maxScore)
      if (alpha >= beta):
        return maxScore
    return maxScore

  def done(self, gameState, depth):
    if (gameState.isWin() or gameState.isLose() or depth == self.depth):
      return True

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    maxScore = float('-inf')
    legalActions = gameState.getLegalActions(0)
    legalActions.remove('Stop')
    a = legalActions[0]
    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      score = self.expectValue(successor, 1, 0)
      if (score > maxScore):
        maxScore = score
        a = action
    return a

  def expectValue(self, gameState, agentID, depth):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID >= gameState.getNumAgents():
      agentID = 0
      return self.maxValue(gameState, agentID, depth+1)

    legalActions = gameState.getLegalActions(agentID)
    scores = []
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      scores.append(self.maxValue(newGameState, agentID+1, depth))
    return sum(scores) / len(scores)

  def maxValue(self, gameState, agentID, depth):
    if (self.done(gameState, depth)):
      return self.evaluationFunction(gameState)

    if agentID != 0:
      return self.expectValue(gameState, agentID, depth)

    maxScore = float('-inf')
    legalActions = gameState.getLegalActions(0)
    for action in legalActions:
      newGameState = gameState.generateSuccessor(agentID, action)
      maxScore = max(maxScore, self.expectValue(newGameState, 1, depth))
    return maxScore

  def done(self, gameState, depth):
    if (gameState.isWin() or gameState.isLose() or depth == self.depth):
      return True
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    things that are important: possible moves(dead end), ghost(distance, scared), food
    Basically, we want to eat the ghosts whenever we can. Since if ghost is not too far, 
    we get huge bonus to score, pacman would try to eat the big dot. if that is not avaliable,
    we check if we have won, if not, we try to eat the closest food. if the distance of that
    is too much(bigger than the depth) we try to go to the first food remaining. Since we use a
    sigmoid function returning a bonus to score of 1 to 9, which is less or equal to a food's worth,
    pacman will try to eat closer food.
  """
  "*** YOUR CODE HERE ***"
  pos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  scared = []
  for scare in scaredTimes:
      if scare > 0:
          scared.append(scare) 
  if len(scared) > 0:
      deliciousGhost = []
      for ghostState in ghostStates:
         if ghostState.scaredTimer>0:
            deliciousGhost.append(mazeDistance(pos, ghostState.getPosition(), currentGameState))
      return currentGameState.getScore() +200/min(deliciousGhost)
  foods = newFood.asList()
  if len(foods)<1:
    return currentGameState.getScore()+10
  import math
  sig = 9/(1 + math.e**(mazeDistance(pos, foods[0], currentGameState))) 
  return currentGameState.getScore()+sig

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

