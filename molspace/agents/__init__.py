"""
This module will be responsible for the different choices of RL agents
including Monte Carlo Tree Search, Hill Climb, A* Searchers, and
Single Shot combinatorial searchers.

The module should plug into an environment interfacing with it's action-step
interface and attempt to maximally explore relevant parts of said environment
with minimum number of search calls.
"""

import molspace.agents.random
import molspace.agents.mcts
import molspace.agents.molwalker
import molspace.agents.lipschitz
