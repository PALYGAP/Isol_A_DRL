# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random


#from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches

import loggers as lg
import importlib

from settings import run_folder, run_archive_folder
import initialise
import pickle


lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# If loading an existing neural network, copy the config file to root
#if initialise.INITIAL_RUN_NUMBER != None:
#    copyfile(run_archive_folder  + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py', './config.py')

import config

######## LOAD MEMORIES IF NECESSARY ########

if initialise.INITIAL_MEMORY_VERSION == None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) + '...')
    memory = pickle.load( open( run_archive_folder + env.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(initialise.INITIAL_MEMORY_VERSION).zfill(4) + ".p",   "rb" ) )

######## LOAD MODEL IF NECESSARY ########

# create an untrained neural network objects from the config file
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
best_NN    = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

#If loading an existing neural netwrok, set the weights from that model
if initialise.INITIAL_MODEL_VERSION != None:
    best_player_version  = initialise.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())
#otherwise just ensure the weights on the two players are the same
else:
    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

#copy the config file to the run folder
copyfile('./config.py', run_folder + 'config.py')
#plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes = True)

print('\n')

######## CREATE THE PLAYERS ########

current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
#user_player = User('player1', env.state_size, env.action_size)
iteration = 0

while 1:

    iteration += 1
    importlib.reload(lg)
    importlib.reload(config) #TODO: fix cause has no effect - reload config.py from run directory
    
    print('ITERATION NUMBER ' + str(iteration))
    
    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))

    ######## SELF PLAY ########
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)
    print('\n')

    # TODO : move inside playMatches ?
    memory.clear_stmemory()
    
    if len(memory.ltmemory) >= config.MEMORY_SIZE:

        ######## RETRAINING ########
        print('RETRAINING...')
        current_player.retrainTheCurrentModel(memory.ltmemory)
        print('')

        if iteration % 5 == 0:
            pickle.dump( memory, open( run_folder + "memory/memory" + str(iteration) + ".p", "wb" ) )

        lg.logger_memory.info('====================')
        lg.logger_memory.info('NEW MEMORIES')
        lg.logger_memory.info('====================')
        
        memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))
        
        for s in memory_samp:
            current_move_value, current_move_probs, _ = current_player.get_move_preds(s['state'])
            best_move_value, best_move_probs, _ = best_player.get_move_preds(s['state'])
            current_tile_value, current_tile_probs, _ = current_player.get_move_preds(s['state'])
            best_tile_value, best_tile_probs, _ = best_player.get_tile_preds(s['state'])

            lg.logger_memory.info('MCTS MOVE VALUE FOR %s: %f', s['playerTurn'], s['value'])
            lg.logger_memory.info('CUR MOVE PRED VALUE FOR %s: %f', s['playerTurn'], current_move_value)
            lg.logger_memory.info('BES MOVE PRED VALUE FOR %s: %f', s['playerTurn'], best_move_value)
            s['state'].render(lg.logger_memory)
            #lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']]  )
            #lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  current_probs])
            #lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  best_probs])

            y_dim=env.grid_shape[0]
            x_dim=env.grid_shape[1]

            # Logs for the MOVE half turn
            lg.logger_memory.info('THE MCTS MOVE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x,2)) for x in s['MV'][x_dim * r: (x_dim * r + x_dim)]  ])
            lg.logger_memory.info('CUR PRED MOVE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x,2)) for x in current_move_probs[x_dim * r: (x_dim * r + x_dim)]  ])
            lg.logger_memory.info('BES PRED MOVE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in best_move_probs[x_dim * r: (x_dim * r + x_dim)] ])
            lg.logger_memory.info('')
            lg.logger_memory.info('ID: %s', s['state'].id)
            lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))
            lg.logger_memory.info('\n')

            # Logs for the TILE half turn
            lg.logger_memory.info('THE MCTS TILE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x,2)) for x in s['TV'][x_dim * r: (x_dim * r + x_dim)]  ])
            lg.logger_memory.info('CUR PRED TILE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x,2)) for x in current_tile_probs[x_dim * r: (x_dim * r + x_dim)]  ])
            lg.logger_memory.info('BES PRED TILE VALUES:')
            for r in range(y_dim):
                lg.logger_memory.info(['----' if float("{0:.2f}".format(x)) == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in best_tile_probs[x_dim * r: (x_dim * r + x_dim)] ])
            lg.logger_memory.info('')
            lg.logger_memory.info('ID: %s', s['state'].id)
            lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))
            lg.logger_memory.info('\n')




        ######## TOURNAMENT ########
        print('TOURNAMENT...')
        scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, turns_until_tau0 = 0, memory = None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)
        #print(points)

        print('\n\n')

        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(env.name, best_player_version)

    else:
        print('MEMORY SIZE: ' + str(len(memory.ltmemory)))
