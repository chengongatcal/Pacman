# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision procestate
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
          val = self.values.copy()
          for s in self.mdp.getStates():
            vals = []
            check = self.mdp.isTerminal(s)
            if not check:
              for action in self.mdp.getPossibleActions(s):
                vals.append(self.getQValue(s, action))
              maxVal = max(vals)
              val[s] = maxVal
          self.values = val


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ret = 0
        transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
        for next, prob in transitionStates:
          r = self.mdp.getReward(state, action, next)
          temp = self.discount * self.getValue(next)
          ret += prob * (r + temp)
        return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        dictionary =  util.Counter()
        actions = self.mdp.getPossibleActions(state)
        
        for a in actions:
          action = self.getQValue(state, a)
          dictionary[a] = action
        ret = dictionary.argMax()
        return ret


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision procestate
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        counter = 0
        while counter < self.iterations:
            count = 0
            vals = self.values.copy()
            states = self.mdp.getStates()
            length = len(states)
            
            for status in states:
              lst = []
              isTerminal = self.mdp.isTerminal(status)
              check = counter % length == count
              
              if not isTerminal and check:
                  for a in self.mdp.getPossibleActions(status):
                      lst.append(self.getQValue(status, a))
                  vals[status] = max(lst)
              count += 1
              
              if count == length:
                count = 0
            self.values = vals
            counter += 1
 

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision procestate
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pred = []
        queue = util.PriorityQueue()
        states = self.mdp.getStates()
        
        for s in self.mdp.getStates():
          for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
              transitionStates = self.mdp.getTransitionStatesAndProbs(state, action)
              for t in transitionStates:
                if t[0] == s:
                  if not t[0] in pred:
                    pred.append(state)
 
        for state in pred:
          isTerminal = self.mdp.isTerminal(state)
          if not isTerminal:
            actions = self.mdp.getPossibleActions(state)
            lst = []
            for action in actions:
              qvals = self.getQValue(state, action)
              lst.append(qvals)
            maxVal = max(lst)
            diff = abs(self.values[state] - maxVal)
            queue.update(state, -diff)
        
        for i in range(self.iterations):
          if queue.isEmpty():
            return
          else:
            state = queue.pop()
            isTerminal = self.mdp.isTerminal(state)
            if not isTerminal:
              lst = []
              actions = self.mdp.getPossibleActions(state)
              for action in actions:
                  lst.append(self.getQValue(state, action))
              maxVal = max(lst)
              self.values[state] = maxVal
            
            predecessors = []
            for getstate in self.mdp.getStates():
              for action in self.mdp.getPossibleActions(getstate):
                transitionStates = self.mdp.getTransitionStatesAndProbs(getstate, action)
                for t in transitionStates:
                  if t[0] == state and not getstate in predecessors:
                    predecessors.append(getstate)
            
            for state in predecessors:
              isTerminal = self.mdp.isTerminal(state)
              if not isTerminal:
                maxVal = max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - maxVal)
                if diff > self.theta:
                  queue.update(state, -diff)    




