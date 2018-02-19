# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl


class User():
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, tau):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return (action, pi, value, NN_value)


class Agent():
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name

        self.state_size = state_size
        self.action_size = action_size

        self.cpuct = cpuct

        self.MCTSsimulations = mcts_simulations
        self.model = model

        self.mcts = None
        self.currentRoot = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_move_loss = []
        self.train_tile_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_move_loss = []
        self.val_tile_loss = []

    def act(self, state, tau):
        # Play one turn for a player
        # A player must play one pawn move and one tile removal to complete its turn

        if self.mcts == None or state.id not in self.mcts.tree:
            # Case first move for the player
            self.buildMCTS(state)
            self.mcts.root.type = "tile" #The starting node type is TILE so that a move can be done first
        else:
            self.changeRootMCTS( state )

        ###############
        ## MOVE STEP ##
        ###############

        lg.logger_mcts.info('\n\n')
        lg.logger_mcts.info('***********************')
        lg.logger_mcts.info('****** MOVE STEP ******')
        lg.logger_mcts.info('***********************\n\n')

        #### run the simulation
        for sim in range(self.MCTSsimulations):
            lg.logger_mcts.info('*************************************')
            lg.logger_mcts.info('****** MOVE STEP SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('*************************************')
            self.simulate("move") #The starting node type is always the same for all the SIM for MCTSsimulations sims
            lg.logger_mcts.info('\n')

        #### get move values
        pMv, mvValues = self.getMV(1)  # 1 is a "temperature" param

        ####pick the move
        move, mvValue = self.chooseMove(pMv, mvValues, tau)

        nextState1, _, _, _ = state.makeMove(move)

        NN_mvValue = -self.get_move_preds(nextState1)[0]

        lg.logger_mcts.info('MOVE VALUES...%s', pMv)
        lg.logger_mcts.info('CHOSEN MOVE...%d', move)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', mvValue)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_mvValue)

        ###############
        ## TILE STEP ##
        ###############

        lg.logger_mcts.info('\n\n')
        lg.logger_mcts.info('***********************')
        lg.logger_mcts.info('****** TILE STEP ******')
        lg.logger_mcts.info('***********************\n\n')

        # Once the move is chosen change the root of the MCTS tree to the newstate node
        self.changeRootMCTS( nextState1 )

        #### run the simulation
        for sim in range(self.MCTSsimulations):
            lg.logger_mcts.info('*************************************')
            lg.logger_mcts.info('****** MOVE TILE SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('*************************************')
            self.simulate("tile")
            lg.logger_mcts.info('\n')

        #### get tile values
        pTv, tiValues = self.getTV(1)  # 1 is a "temperature" param

        ####pick the tile to Remove
        tile, tiValue = self.chooseTileToRemove(pTv, tiValues, tau)

        nextState2, _, _, _ = nextState1.removeTile(tile)

        NN_tiValue = -self.get_tile_preds(nextState2)[0]

        lg.logger_mcts.info('TILEtoREMOVE VALUES...%s', pTv)
        lg.logger_mcts.info('CHOSEN TILEtoREMOVE...%d', tile)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', tiValue)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_tiValue)

        return (move, pMv, mvValue, NN_mvValue, tile, pTv, tiValue, NN_tiValue)



    def simulate(self, nextNodeType):

        if not nextNodeType != self.mcts.root.type:
            print("problem simulate() --> not type != self.mcts.root.type")

        ### Log MCTS exploration
        lg.logger_mcts.info( 'ROOT NODE = %s', self.mcts.root.state.id )
        self.mcts.root.state.render( lg.logger_mcts )
        # lg.logger_mcts.info('CURRENT PLAYER = %d', self.mcts.root.state.playerTurn)
        lg.logger_mcts.info( 'CURRENT PLAYER = %s', self.mcts.root.state.pieces[str(self.mcts.root.state.playerTurn)] )

        ##### MOVE THE LEAF NODE
        leaf, value, done, pathToLeaf, nextNodeType = self.mcts.moveToLeaf( nextNodeType )
        leaf.state.render( lg.logger_mcts )

        ##### EVALUATE THE LEAF NODE
        NN_gameState_value = self.evaluateLeaf( leaf, value, done, nextNodeType )

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, NN_gameState_value, pathToLeaf)



    def evaluateLeaf( self, leaf, value, done, nextNodeType ):

        if leaf.type == nextNodeType :
            print("problem evaluateLeaf() --> leaf.type == nextNodeType")
        lg.logger_mcts.info('------EVALUATING LEAF------')

        if done == 0:

            # Determine the allowed actions (move or tileRemoval)
            # and their probs of "goodness" according to the NN model
            if nextNodeType == "move":
                NN_gameState_value, probs, allowedActions = self.get_move_preds(leaf.state)
            else:
                # leaf.type == "tile"
                NN_gameState_value, probs, allowedActions = self.get_tile_preds(leaf.state)

            lg.logger_mcts.info('(*** %s step *** PREDICTED VALUE FOR %s: %f', leaf.type,
                            self.mcts.root.state.pieces[str(leaf.state.playerTurn)], NN_gameState_value)

            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                if  nextNodeType == "move":
                    newState, _, _, error = leaf.state.makeMove(action)
                else:
                    # leaf.type == "tile"
                    newState, _, _, error = leaf.state.removeTile(action)

                if error != 0:
                    print("problem evaluateLeaf() --> Error the action %d of type %s cannot be done", simulationAction, type)

                # NODE CREATION - If node for the newState does not exist --> create one and add it to the tree
                if newState.id not in self.mcts.tree:
                    newNode = mc.Node( newState, nextNodeType )
                    self.mcts.addNode( newNode )
                    lg.logger_mcts.info( 'added *%s* node = %s    with estimated prob = %f', nextNodeType, newNode.id, probs[idx] )
                else:
                    newNode = self.mcts.tree[newState.id]
                    # newNode.type = nextNodeType # A node can already exist and have been reach through an adge of a different type
                    lg.logger_mcts.info('existing *%s* node...%s...', nextNodeType, newNode.id)

                # EDGE CREATION - Create a new edge and add it to the tree
                if newNode.type != nextNodeType :
                    print("problem evaluateLeaf() --> newNode.type != nextNodeType")
                newEdge = mc.Edge(leaf, newNode, probs[idx], action, nextNodeType)
                leaf.edges.append((action, newEdge))

                #if not leaf.type != newNode.type:
                #   print("problem evaluateLeaf --> not leaf.type != newNode.type")

        else:
            # if the game is finished the "value" is the value received in the evaluateLeaf() call and not the one calculated by the evaluateLeaf()
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

        if done == 0:
            meanNN = NN_gameState_value
            return meanNN
        else:
            return value


    def get_move_preds(self, state):
        # Calculates the probabilities of Allowed Moves and Allowed Tiles for a Game State

        # Predicts the leaf with the NN model of the agent (for a player)
        inputToModel = np.array([self.model.convertToModelInput(state)])

        preds = self.model.predict(inputToModel)  ### Model predicts for the 3 heads with inputToModel as input
        value_array = preds[0]
        moves_array = preds[1]
        value = value_array[0]

        # Probabilities for moves
        mvLogits = moves_array[0]
        allowedMoves = state.allowedMoves
        mvMask = np.ones(mvLogits.shape, dtype=bool)
        mvMask[allowedMoves] = False
        mvLogits[mvMask] = -100
        # SOFTMAX
        mvOdds = np.exp(mvLogits)
        moves_probs = mvOdds / np.sum(mvOdds)  ###put this just before the for?

        return ((value, moves_probs, allowedMoves))


    def get_tile_preds(self, state):
        # Calculates the probabilities of Allowed Moves and Allowed Tiles for a Game State

        # Predicts the leaf with the NN model of the agent (for a player)
        inputToModel = np.array([self.model.convertToModelInput(state)])

        preds = self.model.predict(inputToModel)  ### Model predicts for the 3 heads with inputToModel as input
        value_array = preds[0]
        tiles_array = preds[2]
        value = value_array[0]

        # Probabilities for tiles
        tiLogits = tiles_array[0]
        allowedTilesToRemove = state.allowedTilesToRemove
        tiMask = np.ones(tiLogits.shape, dtype=bool)
        tiMask[allowedTilesToRemove] = False
        tiLogits[tiMask] = -100
        # SOFTMAX
        tiOdds = np.exp(tiLogits)
        tiles_probs = tiOdds / np.sum(tiOdds)  ###put this just before the for?

        return ((value, tiles_probs, allowedTilesToRemove))


    def getMV(self, temperature):
        edges = self.mcts.root.edges
        pMv = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pMv[action] = pow(edge.stats['N'], 1 / temperature)
            values[action] = edge.stats['Q']

        pMv = pMv / (np.sum(pMv) * 1.0)
        return pMv, values


    def getTV(self, temperature):
        # TODO: have 2 temperature ?
        edges = self.mcts.root.edges
        pTv = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for tile, edge in edges:
            pTv[tile] = pow(edge.stats['N'], 1 / temperature)
            values[tile] = edge.stats['Q']

        pTv = pTv / (np.sum(pTv) * 1.0)
        return pTv, values


    def chooseMove(self, pi, values, tau):
        if tau == 0:
            moves = np.argwhere(pi == max(pi))
            move = random.choice(moves)[0]
        else:
            move_idx = np.random.multinomial(1, pi)
            move = np.where(move_idx == 1)[0][0]

        value = values[move]

        return move, value


    def chooseTileToRemove(self, pi, values, tau):
        if tau == 0:
            tiles = np.argwhere(pi == max(pi))
            tile = random.choice(tiles)[0]
        else:
            tile_idx = np.random.multinomial(1, pi)
            tile = np.where(tile_idx == 1)[0][0]

        value = values[tile]

        return tile, value


    def retrainTheCurrentModel(self, ltmemory):
        lg.logger_mcts.info('******RETRAINING MODEL******')

        for i in range(config.TRAINING_LOOPS):
            training_set = random.sample(ltmemory, min(config.TRAINING_SET_SIZE, len(ltmemory)))

            training_states = np.array([self.model.convertToModelInput(row['state']) for row in training_set])
            training_targets = {'value_head': np.array([row['value'] for row in training_set])
                , 'move_head': np.array([row['MV'] for row in training_set])
                , 'tile_head': np.array([row['TV'] for row in training_set])}

            fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=config.MINI_BATCH_SIZE)
            lg.logger_mcts.info('NEW LOSS %s', fit.history)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_move_loss.append(round(fit.history['move_head_loss'][config.EPOCHS - 1], 4))
            self.train_tile_loss.append(round(fit.history['tile_head_loss'][config.EPOCHS - 1], 4))

        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_move_loss, 'k--')
        plt.plot(self.train_tile_loss, 'k--')

        plt.legend(['train_overall_loss', 'train_value_loss', 'train_move_loss', 'train_tile_loss'], loc='lower left')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(5.0)

        print('\n')
        self.model.printWeightAverages()


    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds


    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.currentRoot = mc.Node(state, "move")
        self.mcts = mc.MCTS(self.currentRoot, self.cpuct)


    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT (move Node) OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
        self.mcts.root = self.mcts.tree[state.id]
