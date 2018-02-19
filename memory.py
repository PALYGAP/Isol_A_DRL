import numpy as np
from collections import deque

import config

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	def commit_stmemory(self, state, moveValues, tileValues):
		# Add an element to the short term memory
		self.stmemory.append({
				  'board': state.board
				, 'state': state
				, 'id': state.id
				, 'MV': moveValues
				, 'TV': tileValues
				, 'playerTurn': state.playerTurn
				})

	def commit_ltmemory(self):
		# Save short term memories to long term memories
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)
		