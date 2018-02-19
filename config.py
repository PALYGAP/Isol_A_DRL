#### ISOLA GAME
GAME_HEIGTH = 6 # Y axis - Even value is 'normal' to have the players start on different rows
GAME_WIDTH = 8 # X axis
PLAYER_1 = 9 # the player that starts the game
PLAYER_2 = -9
HAlF_STEP_MOVE = 1
HAlF_STEP_TILE = 2
LOOKED_TILE_1 = int(GAME_WIDTH * ((GAME_HEIGTH / 2) - 1))
LOOKED_TILE_2 = int((GAME_WIDTH * ((GAME_HEIGTH / 2) + 1)) - 1)


#### SELF PLAY
EPISODES = 5          	# Number of self play before retraining the model
MCTS_SIMS = 100        	# Number of MCTS simutations from one game state ( = number of node explored )
MEMORY_SIZE = 250     	# Number of memories (training sample) keep for the model retraining
TURNS_UNTIL_TAU0 = 10 	# turn on which it starts playing deterministically
CPUCT = 1			  	# in MCTS, the constant determining the level of exploration
EPSILON = 0.2			# in MCTS, used if the exploration is at the root node. Lead to different formula of U term
ALPHA = 0.8


#### RETRAINING (and first Training)
TRAINING_SET_SIZE = 256  # Training set size used during re-training - Sampled from the LT Memories
EPOCHS = 3			     # Number of full presentation of the training set
MINI_BATCH_SIZE = 32     # Size of the mini-Batch
REG_CONST = 0.0001	     # Regulizer coefficient of the l2 regulizer of KERAS
LEARNING_RATE = 0.1		 # Learning rate of the SGD
MOMENTUM = 0.9			 # Momentum of the SGD
TRAINING_LOOPS = 10		 # Number of time the networks is trained with a different randomly choosen training set (sampled from LT memories)

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3



"""KERAS Stochastic gradient descent optimizer.

Includes support for momentum,
learning rate decay, and Nesterov momentum.

# Arguments
    lr: float >= 0. Learning rate.
    momentum: float >= 0. Parameter that accelerates SGD
        in the relevant direction and dampens oscillations.
    decay: float >= 0. Learning rate decay over each update.
    nesterov: boolean. Whether to apply Nesterov momentum.
"""




HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]


