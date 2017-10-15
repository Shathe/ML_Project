#!/usr/bin/env python
from tensorflow.contrib.keras  import backend as K
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing
import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun # whichever is imported "as game" will be used
import dummy_game
import tetris_fun as game
import random
import numpy as np
from collections import deque
import time
GAME = 'tetris' # the name of the game being played for log files
ACTIONS = 6 # number of valid actions:rotate_iz, rotate_der, iz, der, not do, down
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training: just observing, gamma not goind down 
EXPLORE = 500. # frames over which to anneal epsilon: frames between gamma=max and gamma=min.
#Training. When 500 frames are explore, training keeps but gamma stays at 0.05
FINAL_EPSILON = 0.05 # final value of epsilon. Explore vs explote of q-learning
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 64])
    b_conv1 = bias_variable([64])

    W_conv2 = weight_variable([4, 4, 64, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    
    W_fc1 = weight_variable([3200, 1024])
    b_fc1 = bias_variable([1024])

    W_fc2 = weight_variable([1024, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 3200])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

'''
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3

    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x) # use_bias=False,
    x = layers.BatchNormalization(axis=bn_axis)(x) # scale=False,
    x = layers.Activation('relu')(x)
    return x

def createNetwork2():
    # network weights
    #conv2d_bn(tower_2, nb_filter, 5, 5, padding='same', strides=(1, 1))

    inputs = layers.Input(shape=(80, 80, 4))

    # a layer instance is callable on a tensor, and returns a tensor
    x = layers.Conv2D(32, (8, 8), padding='valid', activation='relu', input_shape=input_shape,
                     kernel_initializer='truncated_normal', strides=(4, 4))(inputs)

    x = conv2d_bn(x, 64, 4, 4, padding='valid', strides=(2, 2))

    x = conv2d_bn(x, 128, 3, 3, padding='valid', strides=(1, 1))
    x = layers.Flatten()(x)
    h_fc1 = layers.Dense(512, activation='softmax')(x)
    predictions = layers.Dense(ACTIONS, activation='softmax')(h_fc1)

    return inputs, predictions, h_fc1
'''
def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS]) #placeholder actions. Vector que tendra el Q valor
    y = tf.placeholder("float", [None]) # placeholder who really maxs la accion. Seraa el Q valor
    # readout, is the output of the network
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1) #Q valor de acciones * acciones elegidas -> Q valor de accions elegidas
    #Entrenas para que readout sea el valor Q
    cost = tf.reduce_mean(tf.square(y - readout_action)) 
    #El coste es el cuadrado de la resta del valor Q (el valor de la accion, cuanto mayor mejor) de la verdadera, menos el valor Q de la que has elegido
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1 # Primera acion sera no hacer nada, que es el primer valor
    x_t, r_0, terminal = game_state.frame_step(do_nothing) # Cuadno el das una accion al juego, te da el estado (el siguiente frame, la recompensa y si es terminal)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY) #x_T es el frame hecho resize y en blanco y negro
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    # s_t es el stack de 4 frames que tienes que ar de input pero al princpio no tienes y tienes que pasarle 4 iguales

    # saving and loading networks
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()


    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"
	start = time.time()
    epsilon = INITIAL_EPSILON
    t = 0 #timestep=frames
    while "pigs" != "fly":

        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0] #evaluas la red sobre el primer input y te da el valor esprado
        a_t = np.zeros([ACTIONS]) #accion a tomar
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE: # EXPLORAR
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else: # EXPLOTACION
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # Aqui esta a_t, la accion a tomar [0,0,1,0,0,0] cone xplotacion o exploracion segun el valor de epsilon
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            #si estamos en observacion, ir decremtnando epsilon cada vez mas

        for i in range(0, K):#Esto tendria que ser solo un paso, es aplicar K veces la accion seleccionada
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

           	# AHORA TENEMOS NUEVO s_t1 PARA EL SIGUIENTE ESTADO

            # store the transition in D, DE LA CUAL SACARAS AL AZAR

            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # only train if done observing SI ESTAS EXPLORANDO O ENTRENANADO, 
        if t > OBSERVE:
            # sample a minibatch to train on (MINIBATCH DE LA MEMORIA)
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch] # el s_t
            a_batch = [d[1] for d in minibatch] # el a_t (accion)
            r_batch = [d[2] for d in minibatch] # r_t :reward
            s_j1_batch = [d[3] for d in minibatch] # s_t1 : resultado despues de ejecutarlo

            y_batch = [] #L
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):# CALCULAR VALOR Y (RECOMPENSA)
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch, # si es estado final, y=recompensa del estado final. si no, y= recomepnsa de ese estado +GAMMA*recompensamaxima del siguiente estado
                a : a_batch, # Accion a tomar (0,0,0,1,0,0). La que ha dado el sampleo de minibaatch
                s : s_j_batch}) #s: input images, las que ha dado directamente el sampleo de minibatch

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        if t % 1000 == 0:
        	print(time.time()-start)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print "TIMESTEP", t, "/ STATE", state, "/ LINES", game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
