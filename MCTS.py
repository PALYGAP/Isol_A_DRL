import numpy as np
import logging
import config

from utils import setup_logger
import loggers as lg


class Node():

    def __init__(self, state, type):
        self.state = state
        self.playerTurn = state.playerTurn
        self.id = state.id
        self.edges = []
        self.type = type

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge():

    def __init__(self, inNode, outNode, prior, action, type):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn
        self.action = action
        self.type = type

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


class MCTS():

    def __init__(self, root, cpuct):
        self.root = root  # Node object ( contains game state)
        self.tree = {}
        self.cpuct = cpuct  # Coeff of U (exploration contribution)
        self.addNode(root)  # Add the first node to the tree an initize it with" root"

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self, nextNodeTypeRec):
        nextNodeType = nextNodeTypeRec
        lg.logger_mcts.info('------MOVING TO LEAF------')

        pathToLeaf = []
        currentNode = self.root
        oldType = ""

        done = 0
        leafValue = 0

        while not currentNode.isLeaf():
            # Browse the tree upward to a leaf
            # At each non-leaf node choose edge to foolow accorting to maxQU (QU is calculated for all the "upward unexplored" edge

            if currentNode.edges[0][1].type != nextNodeType :
                print("problem moveToLeaf() --> currentNode.edges[0][1].type != nextNodeType")

            lg.logger_mcts.info('TURN of PLAYER   %s', self.root.state.pieces[str(currentNode.state.playerTurn)])

            maxQU = -99999

            if currentNode == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            # Calculate Nb for the current node (sum of Nb on each upward edge
            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            # Work out which edge has the max Q + U value
            for idx, (action, edge) in enumerate(currentNode.edges):
                # Apply coeedficient to the eploration term
                # the term  " + epsilon * nu[idx]" = 0 apart for "root" node
                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                # lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
                # 	, action, action % 7, edge.stats['N'], round(edge.stats['P'],6), round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
                # 	, round(edge.stats['W'],6), round(Q,6), round(U,6), round(Q+U,6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge
                    if simulationEdge.type != nextNodeType :
                        print("problem moveToLeaf() --> simulationEdge.type != nextNodeTyp")

            lg.logger_mcts.info('max(Q + U) %s = %d', simulationEdge.type, simulationAction)
            if simulationEdge.type != nextNodeType :
                print("problem moveToLeaf() --> simulationEdge.type != nextNodeType")
                break
            if simulationEdge.type == oldType :
                print("problem moveToLeaf() --> simulationEdge.type == oldType")
                break
            oldType = simulationEdge.type

            # Calculate the value of the newState from the POV of the new playerTurn
            # and check if the game is finished

            #if nextNodeType != edge.type:
            #    print("problem moveToLeaf() -->: nextNodeType != simulationAction.type")

            if nextNodeType == 'move':
                _, leafValue, done, error = currentNode.state.makeMove(
                    simulationAction)  # the value of the newState from the POV of the new playerTurn
                nextNodeType = 'tile'
            else:
                # case simulationEdge.outnode.type == 'tile'
                _, leafValue, done, error = currentNode.state.removeTile(
                    simulationAction)  # the value of the newState from the POV of the new playerTurn
                nextNodeType = 'move'

            if error != 0:
                print(" problem moveToLeaf() --> Error the action %d of type %s cannot be done", simulationAction, type )

            # Update currentNode an add the chossen edge to the pathToLeaf
            lastNode = currentNode
            currentNode = simulationEdge.outNode
            #if lastNode.type == currentNode.type:
            #    print("problem: lastNode.type == currentNode.type")

            # Update the path to leaf
            pathToLeaf.append(simulationEdge)

        # Log if the simulation reached the end of the game
        lg.logger_mcts.info('GAME Finished = %s', 'TRUE' if done == 1 else 'FALSE')

        if currentNode.type == nextNodeType :
            print("problem moveToLeaf() --> currentNode.type == nextNodeType")

        return currentNode, leafValue, done, pathToLeaf, nextNodeType


    def backFill(self, leaf, value, pathToLeaf):
        lg.logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.state.playerTurn

        for edge in pathToLeaf:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            lg.logger_mcts.info('updating edge with value %f for player %d  -  N = %d, W = %f, Q = %f'
                                , value * direction
                                , playerTurn
                                , edge.stats['N']
                                , edge.stats['W']
                                , edge.stats['Q']
                                )

            edge.outNode.state.render(lg.logger_mcts)


    def addNode(self, node):
        self.tree[node.id] = node
