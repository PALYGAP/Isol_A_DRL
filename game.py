import numpy as np
# import logging
import config


class Game:
    def __init__(self):
        self.grid_shape = (config.GAME_HEIGTH, config.GAME_WIDTH)
        # 2D conv - There is only convolution applied to GAME_HEIGTH, GAME_WIDTH dim
        # Only 2 slices of inputs : players_position and tiles_state
        self.input_shape = (2, config.GAME_HEIGTH, config.GAME_WIDTH)

        # Set the initial position of the players' pieces
        self.gameState = self.reset()
        self.actionSpace = np.full((config.GAME_HEIGTH * config.GAME_WIDTH,), 1)  # TODO remove

        self.name = 'Isol-A'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(
        self.actionSpace)  # TODO replace with --> self.action_size = config.GAME_HEIGTH * config.GAME_WIDTH

    def reset(self):
        # Set the initial position of the players' pieces
        initBoard = np.full((config.GAME_HEIGTH * config.GAME_WIDTH), 1)
        initBoard[ config.LOOKED_TILE_1 ] = config.PLAYER_1
        initBoard[ config.LOOKED_TILE_2 ] = config.PLAYER_2
        self.currentPlayer = config.PLAYER_1 # the player codification :  9 first_player  -9 second player
        self.halfStep = config.HAlF_STEP_MOVE # Codification 1 for Move and 2 for tile
        self.gameState = GameState(initBoard, self.currentPlayer, self.halfStep )  # 9 for 1st Player

        return self.gameState

    def step(self, move, tile):
        next_state1, value1, done1, _ = self.gameState.makeMove(move)
        if done1 == 0:
            next_state2, value2, done2, _ = next_state1.removeTile(tile)
            self.gameState = next_state2
        else:
            # Case where the game ends after the move (no need to remove a tile)
            self.gameState = next_state1

        self.currentPlayer = -self.currentPlayer
        info = None

        if done1 == 0:
            return((self.gameState, value2, done2, info))
        else:
            return ((self.gameState, value1, done1, info))

class GameState():
    def __init__(self, board, playerTurn, halfStep):
        #Tile looked during the game
        self.lookedTile1 = int(   config.GAME_WIDTH *  ( (config.GAME_HEIGTH/2) -1)  )
        self.lookedTile2 = int(  (config.GAME_WIDTH * ( (config.GAME_HEIGTH/2) +1) ) - 1  )

        self.board = board  # the board codification : 0 no_tile   1 tile_present   9 first_player   -9 second_Player
        self.pieces = {'1': '#', '0': ' ', '9': '1', '-9': '2'}
        self.playerTurn = playerTurn  # the player codification :  9 first_player  -9 second player
        self.halfStep = halfStep # Codification 1 for Move and 2 for til
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedMoves = self._allowedMoves()
        self.allowedTilesToRemove = self._allowedTilesToRemove()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedMoves(self):
        # Identifies the position of the current player
        allowedMoves = []
        pos_current_player = -9999
        for i in range(len(self.board)):
            if self.board[i] == self.playerTurn:
                pos_current_player = i

        # Check for a free position surrounding the Current Player position
        x_pos = pos_current_player % config.GAME_WIDTH
        y_pos = int(pos_current_player / config.GAME_WIDTH)

        for j in range(x_pos - 1, x_pos + 2):
            for k in range(y_pos - 1, y_pos + 2):
                if not (j < 0 or j > config.GAME_WIDTH - 1 or k < 0 or k > config.GAME_HEIGTH - 1):
                    pos = k * config.GAME_WIDTH + j
                    if self.board[ pos ] == 1:  # if the opponent is next to the current player --> not allowed cause = 9 or = -9 and player can't stay put
                        allowedMoves.append( pos )
        return allowedMoves

    def _allowedTilesToRemove(self):
        # Identifies the tiles that can be removed --> all positions that have a tile apart from the stating tiles !!!
        allowedTiles = []
        for i in range(len(self.board)):
            if self.board[i] == 1 and ( i != config.LOOKED_TILE_1 and   i !=  config.LOOKED_TILE_2 ):
                allowedTiles.append(i)
        return allowedTiles

    def _binary(self):
        # Generate the input of the NN
        tiles_state = np.zeros(len(self.board), dtype=np.int)
        players_position = np.zeros(len(self.board), dtype=np.int)
        # second_player_pos = np.zeros(len(self.board), dtype=np.int)

        for i in range(config.GAME_WIDTH * config.GAME_HEIGTH):
            if self.board[i] != 0:
                tiles_state[i] = 1

            if self.board[i] == config.PLAYER_1:
                if self.playerTurn == config.PLAYER_1:
                    players_position[i] = 1
                else:
                    players_position[i] = -1

            if self.board[i] == config.PLAYER_2:
                if self.playerTurn == config.PLAYER_2:
                    players_position[i] = 1
                else:
                    players_position[i] = -1

        position = np.append(tiles_state, players_position)

        return (position)

    def _convertStateToId(self):
        id = ''.join(map(str, self.board)) + str( self.playerTurn ) + str( self.halfStep )
        # TODO: Check if there isn't a problem with the id being fixed length ?
        return id

    def _checkForEndGame(self):
        # Check if the game if finished
        if len(self.allowedMoves) == 0:
            return 1
        else:
            return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if len(self.allowedMoves) == 0:
            return (-1, -1, 1)
        else:
            return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def makeMove(self, move):
        newBoard = np.array(self.board)
        error = 1

        # TODO: Check if move is allowed

        # Set the current player (playerTrurn) previous position to 1 -> tile present
        for i in range( len(self.board) ):
            if self.board[ i ] == self.playerTurn:
                newBoard[ i ] = 1
                error = 0
                break

        # Set new position of the current player
        if self.board[ move ] == 1:
            newBoard[move] = self.playerTurn
        else:
            error = error + 2

        # Build new GameState
        newState = GameState( newBoard, self.playerTurn, config.HAlF_STEP_MOVE ) # After making a pawn move it is still the current player turn (to remove a tile)

        # Check for end of game after move --> It can happen that the other player is isolated even before he current player removes a tile
        value = 0
        done = 0
        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done, error)

    def removeTile(self, tile):
        newBoard = np.array(self.board)
        error = 0

        # TODO: Check if tile removal is allowed
        if self.board[ tile ] == 1:
            newBoard[tile] = 0
            newState = GameState(newBoard, -self.playerTurn, config.HAlF_STEP_TILE)
        else:
            error = 1
            return ( _, _, _, error )

        # Check for end of game after tile removal --> more usual way to end game
        value = 0
        done = 0
        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done, error)

    def render(self, logger):
        for r in range(config.GAME_HEIGTH):
            logger.info([self.pieces[str(x)] for x in
                         self.board[config.GAME_WIDTH * r:   (config.GAME_WIDTH * r + config.GAME_WIDTH)]])
        logger.info('--------------')
