import gym
import itertools
import matplotlib
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import defaultdict
from scipy.stats import beta
from scipy.stats import t
from random import shuffle
from lib import plotting
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

num_action = 13
#num_state_elements = 6    #num_state_elements = 6
#num_state = 2**num_state_elements     #num_state = 100
#value_state = defaultdict(lambda: np.zeros(num_state_elements))
space_action = range(num_action)

train_size = 1
num_size = 1000
window_num = 100

mean = [50, 80]
cov = [[2, 0], [0, 2]]  # diagonal covariance


def Env_calculate_transition_prob(i_sample, current_s, action, ordinal_error, calculate_error,
    value_ha, value_hb, value_hdh, value_hdv, value_va, value_vb, value_vdh, value_vdv,
     present_order, current_order, is_hv):
    
    tolerance_p = 0.02
    #tolerance_v = 8.8
    tolerance_v = 0.02
    #tolerance_v = 0.1
    value_new_state = np.array(current_s)
    value_e_a = np.multiply(value_ha, value_va)
    value_e_b = np.multiply(value_hb, value_vb)
    value_e_dh = np.multiply(value_hdh, value_vdh)
    value_e_dv = np.multiply(value_hdv, value_vdv)

    guass_mu = 0
    calculate_noise = np.random.normal(guass_mu, calculate_error, 1)  
    ordinal_probs = np.array([ordinal_error, 1 - ordinal_error])
    ordinal_obs = np.array([1, 2, 3])
    is_correct = np.random.choice(np.array([0, 1]), p=ordinal_probs)

    if is_hv == 0:
        value_hd = value_hdh
        value_vd = value_vdh
        value_e_d = value_e_dh
    else:
        value_hd = value_hdv
        value_vd = value_vdv
        value_e_d = value_e_dv

    if action == space_action[0] and value_new_state[9]!=0 and value_new_state[10]!=0:
        if is_correct == 0:
            value_new_state[0] = np.random.choice(ordinal_obs)
        else:
            if (value_ha[i_sample] > value_hb[i_sample] + tolerance_p):
                value_new_state[0] = 1
            elif (value_ha[i_sample] < value_hb[i_sample] + tolerance_p) and (value_ha[i_sample] > value_hb[i_sample] - tolerance_p):
                value_new_state[0] = 2
            else:
                value_new_state[0] = 3
    elif action == space_action[1] and value_new_state[9]!=0 and value_new_state[11]!=0:
        if is_correct == 0:
            value_new_state[1] = np.random.choice(ordinal_obs)
        else:
            if (value_ha[i_sample] > value_hd[i_sample] + tolerance_p):
                value_new_state[1] = 1
            elif (value_ha[i_sample] < value_hd[i_sample] + tolerance_p) and (value_ha[i_sample] > value_hd[i_sample] - tolerance_p):
                value_new_state[1] = 2
            else:
                value_new_state[1] = 3
    elif action == space_action[2] and value_new_state[10]!=0 and value_new_state[11]!=0:
        if is_correct == 0:
            value_new_state[2] = np.random.choice(ordinal_obs)
        else:
            if (value_hb[i_sample] > value_hd[i_sample] + tolerance_p):
                value_new_state[2] = 1
            elif (value_hb[i_sample] < value_hd[i_sample] + tolerance_p) and (value_hb[i_sample] > value_hd[i_sample] - tolerance_p):
                value_new_state[2] = 2
            else:
                value_new_state[2] = 3
    elif action == space_action[3] and value_new_state[9]!=0 and value_new_state[10]!=0:
        if is_correct == 0:
            value_new_state[3] = np.random.choice(ordinal_obs)
        else:
            if (value_va[i_sample] > value_vb[i_sample] + tolerance_v):
                value_new_state[3] = 1
            elif (value_va[i_sample] < value_vb[i_sample] + tolerance_v) and (value_va[i_sample] > value_vb[i_sample] - tolerance_v):
                value_new_state[3] = 2
            else:
                value_new_state[3] = 3
    elif action == space_action[4] and value_new_state[9]!=0 and value_new_state[11]!=0:
        if is_correct == 0:
            value_new_state[4] = np.random.choice(ordinal_obs)
        else:
            if (value_va[i_sample] > value_vd[i_sample] + tolerance_v):
                value_new_state[4] = 1
            elif (value_va[i_sample] < value_vd[i_sample] + tolerance_v) and (value_va[i_sample] > value_vd[i_sample] - tolerance_v):
                value_new_state[4] = 2
            else:
                value_new_state[4] = 3
    elif action == space_action[5] and value_new_state[10]!=0 and value_new_state[11]!=0:
        if is_correct == 0:
            value_new_state[5] = np.random.choice(ordinal_obs)
        else:
            if (value_vb[i_sample] > value_vd[i_sample] + tolerance_v):
                value_new_state[5] = 1
            elif (value_vb[i_sample] < value_vd[i_sample] + tolerance_v) and (value_vb[i_sample] > value_vd[i_sample] - tolerance_v):
                value_new_state[5] = 2
            else:
                value_new_state[5] = 3
    elif action == space_action[9] and value_new_state[9]!=0:
        value_new_state[6] = np.around(value_ha[i_sample] * value_va[i_sample] + calculate_noise, 1)
    elif action == space_action[10] and value_new_state[10]!=0:
        value_new_state[7] = np.around(value_hb[i_sample] * value_vb[i_sample] + calculate_noise, 1)
    elif action == space_action[11] and value_new_state[11]!=0:
        value_new_state[8] = np.around(value_hd[i_sample] * value_vd[i_sample] + calculate_noise, 1)
    elif action == space_action[12]:       
        if current_order >= 3:
            current_order = 3
        else:
            current_order += 1
        idx1 = np.where(present_order == current_order)
        int_idx1 = int(idx1[0]) + 9 
        value_new_state[int_idx1] = int(present_order[int(idx1[0])])  # value_new_state[int_idx1] = current_order

    else:
        value_new_state = np.array(current_s)


#    new_state =  int(value_new_state[0]*4**5 + value_new_state[1]*4**4 +
#                  value_new_state[2]*4**3 + value_new_state[3]*4**2 +
#                  value_new_state[4]*4**1 + value_new_state[5])
   
    maximum_ev = np.max([value_e_a[i_sample], value_e_b[i_sample], value_e_d[i_sample]])
    if action == space_action[6]:
        is_done = True
        reward = 10 if value_e_a[i_sample] == maximum_ev else -10
        #reward = value_e_a[i_sample]
        #reward = np.random.choice([value_va[i_sample], 0], p=[value_ha[i_sample], 1 - value_ha[i_sample]])
    elif action == space_action[7]:
        is_done = True
        reward = 10 if value_e_b[i_sample] == maximum_ev else -10
        #reward = value_e_b[i_sample]
        #reward = np.random.choice([value_vb[i_sample], 0], p=[value_hb[i_sample], 1 - value_hb[i_sample]])
    elif action == space_action[8]:
        is_done = True
        reward = 10 if value_e_d[i_sample] == maximum_ev else -10
        #reward = value_e_d[i_sample]
        #reward = np.random.choice([value_vd[i_sample], 0], p=[value_hd[i_sample], 1 - value_hd[i_sample]])
    else:
        is_done = False
        reward = -1
    
    #reward = 100.0 if (new_state == num_state - 1) else -1.0
    return tuple(value_new_state), reward, is_done, current_order


def make_epsilon_greedy_policy(Q, nA):
    def policy_fn(observation, epsilon):
        #'''
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[tuple(observation)])
        A[best_action] += (1.0 - epsilon)
        return A
        #'''
        '''
        value_current_state = np.array(value_state[observation])
        available_action = [ac for ac in range(num_state_elements) if value_current_state[ac] == 0] + [6, 7, 8]
        nA = len(available_action)
        A = np.zeros(num_action)
        A[available_action] = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
        '''
    return policy_fn


def test_accuracy(ordinal_error, calculate_error):
    test_size = 100
    '''
    test_v_a = np.zeros(test_size)
    test_v_b = np.zeros(test_size)
    test_v_d = np.zeros(test_size)
    test_p_a = np.zeros(test_size)
    test_p_b = np.zeros(test_size)
    test_p_d = np.zeros(test_size)
    '''
    #for i_test in range(test_size):
        #test_v_a[i_test], test_v_b[i_test], test_v_d[i_test], test_p_a[i_test], test_p_b[i_test], test_p_d[i_test] = GenerateData()
    
    value_ha, value_va = np.random.multivariate_normal(mean, cov, test_size).T
    random_number_s = np.random.choice(np.linspace(-2., 2., 101), test_size)
    value_e_a = np.multiply(value_ha, value_va)
    value_hb = value_va + random_number_s
    value_vb = value_e_a / value_hb

    random_number_DH = np.random.choice(np.linspace(7, 9, 101), test_size)
    value_hdh = value_ha - random_number_DH
    value_vdh = value_va
    random_number_DV = np.random.choice(np.linspace(7, 9, 101), test_size)
    value_hdv = value_hb
    value_vdv = value_vb - random_number_DV

    value_ha = np.round(value_ha, 2)
    value_hb = np.round(value_hb, 2)
    value_hdh = np.round(value_hdh, 2)
    value_vdh = np.round(value_vdh, 2)
    value_va = np.round(value_va, 2)
    value_vb = np.round(value_vb, 2)
    value_vdv = np.round(value_vdv, 2)
    value_hdv = np.round(value_hdv, 2)
    value_e_a = np.multiply(value_ha, value_va)
    value_e_b = np.multiply(value_hb, value_vb)
    value_e_dh = np.multiply(value_hdh, value_vdh)
    value_e_dv = np.multiply(value_hdv, value_vdv)

    choice_calculate = np.zeros(3)
    choice_poT = np.zeros(6)
    choice_poC = np.zeros(6)
    choice_true = 0
    calculative_reward = 0
    rational_choice = np.zeros(3)
    #test_size = len(test_v_a)
    for i_test in range(test_size):
        state = (0, 0, 0, 0, 0, 0, -100, -100, -100, 0, 0, 0)
        done = 0
        action = 0      
        current_order = 0
        present_order = [i for i in np.array([1,2,3])]
        shuffle(present_order)
        present_order = np.array(present_order)
        is_hv = np.random.choice(np.array([0, 1]), p=[0.5,0.5])
        for t in itertools.count():

            policy = make_epsilon_greedy_policy(Q, num_action) 
            action_probs = policy(state, 0.0)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, current_order =  Env_calculate_transition_prob(i_test, state, action, ordinal_error, calculate_error,
            value_ha, value_hb, value_hdh, value_hdv, value_va, value_vb, value_vdh, value_vdv, present_order, current_order, is_hv)
            calculative_reward += reward

            if done or t == 20:
                break
        
            state = next_state
        best_action = action

        if is_hv == 0:
            value_hd = value_hdh
            value_vd = value_vdh
            value_e_d = value_e_dh
            if np.array_equal(present_order, [1,2,3]):
                if best_action == 6:
                    choice_poT[0] += 1
                elif best_action == 7:
                    choice_poC[0] += 1
            elif np.array_equal(present_order, [2,1,3]):
                if best_action == 6:
                    choice_poT[1] += 1
                elif best_action == 7:
                    choice_poC[1] += 1
            elif np.array_equal(present_order, [1,3,2]):
                if best_action == 6:
                    choice_poT[2] += 1
                elif best_action == 7:
                    choice_poC[2] += 1
            elif np.array_equal(present_order, [2,3,1]):
                if best_action == 6:
                    choice_poT[3] += 1
                elif best_action == 7:
                    choice_poC[3] += 1
            elif np.array_equal(present_order, [3,1,2]):
                if best_action == 6:
                    choice_poT[4] += 1
                elif best_action == 7:
                    choice_poC[4] += 1
            elif np.array_equal(present_order, [3,2,1]):
                if best_action == 6:
                    choice_poT[5] += 1
                elif best_action == 7:
                    choice_poC[5] += 1
        else:
            value_hd = value_hdv
            value_vd = value_vdv
            value_e_d = value_e_dv
            if np.array_equal(present_order, [1,2,3]):
                if best_action == 7:
                    choice_poT[0] += 1
                elif best_action == 6:
                    choice_poC[0] += 1
            elif np.array_equal(present_order, [2,1,3]):
                if best_action == 7:
                    choice_poT[1] += 1
                elif best_action == 6:
                    choice_poC[1] += 1
            elif np.array_equal(present_order, [1,3,2]):
                if best_action == 7:
                    choice_poT[2] += 1
                elif best_action == 6:
                    choice_poC[2] += 1
            elif np.array_equal(present_order, [2,3,1]):
                if best_action == 7:
                    choice_poT[3] += 1
                elif best_action == 6:
                    choice_poC[3] += 1
            elif np.array_equal(present_order, [3,1,2]):
                if best_action == 7:
                    choice_poT[4] += 1
                elif best_action == 6:
                    choice_poC[4] += 1
            elif np.array_equal(present_order, [3,2,1]):
                if best_action == 7:
                    choice_poT[5] += 1
                elif best_action == 6:
                    choice_poC[5] += 1

        if best_action == 6:
            choice_calculate[0] += 1
        elif best_action == 7:
            choice_calculate[1] += 1
        elif best_action == 8:
            choice_calculate[2] += 1

        max_EV = np.argmax([value_e_a[i_test], value_e_b[i_test], value_e_d[i_test]])
        if max_EV == 0:
            rational_choice[0] += 1
        elif max_EV == 1:
            rational_choice[1] += 1
        elif max_EV == 2:
            rational_choice[2] += 1

        if best_action == (max_EV + 6): choice_true += 1

    choice_accuracy = choice_true / test_size
    choice_calculate = choice_calculate / test_size
    rational_choice = rational_choice / test_size
    choice_poT = choice_poT / test_size
    choice_poC = choice_poC / test_size

    return choice_accuracy, calculative_reward, choice_calculate, rational_choice, choice_poT, choice_poC


def q_learning(num_episodes, discount_factor=0.9, alpha=0.1, ordinal_error = 0.3, calculate_error = 0.5,
               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=500):
    #Q = defaultdict(lambda: np.zeros(num_action))

    # Keeps track of useful statistics
    
    cal_num = num_episodes
    converge_num = num_episodes - cal_num
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_maxrewards=np.zeros(num_episodes),
        episode_accuracy=np.zeros(cal_num),
        episode_test_rewards=np.zeros(cal_num),
        episode_choice_calculate=np.zeros((cal_num, 3)),
        episode_rational_choice=np.zeros((cal_num, 3)),
        episode_choice_poT=np.zeros((cal_num, 6)),
        episode_choice_poC=np.zeros((cal_num, 6)))
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(Q, num_action)
    epsilon = epsilon_start

    for i_episode in range(num_episodes):
        
        value_ha, value_va = np.random.multivariate_normal(mean, cov, num_size).T
        random_number_s = np.random.choice(np.linspace(-2., 2., 101), num_size)
        value_e_a = np.multiply(value_ha, value_va)
        value_hb = value_va + random_number_s
        value_vb = value_e_a / value_hb

        random_number_DH = np.random.choice(np.linspace(7, 9, 101), num_size)
        value_hdh = value_ha - random_number_DH
        value_vdh = value_va
        random_number_DV = np.random.choice(np.linspace(7, 9, 101), num_size)
        value_hdv = value_hb
        value_vdv = value_vb - random_number_DV

        value_ha = np.round(value_ha, 2)
        value_hb = np.round(value_hb, 2)
        value_hdh = np.round(value_hdh, 2)
        value_vdh = np.round(value_vdh, 2)
        value_va = np.round(value_va, 2)
        value_vb = np.round(value_vb, 2)
        value_vdv = np.round(value_vdv, 2)
        value_hdv = np.round(value_hdv, 2)
        value_e_a = np.multiply(value_ha, value_va)
        value_e_b = np.multiply(value_hb, value_vb)
        value_e_dh = np.multiply(value_hdh, value_vdh)
        value_e_dv = np.multiply(value_hdv, value_vdv)

        max_EV = np.zeros(num_size)
            
        if (i_episode + 1) % 100 == 0:
            print("\rEpisodes {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush() 
        t_length = 0

        for i_sample in range(num_size):

            state = (0, 0, 0, 0, 0, 0, -100, -100, -100, 0, 0, 0)
            present_order = [i for i in np.array([1,2,3])]
            current_order = 0
            shuffle(present_order)
            present_order = np.array(present_order)
            is_hv = np.random.choice(np.array([0, 1]), p=[0.5,0.5])
            max_EV[i_sample] = np.max([value_e_a[i_sample], value_e_b[i_sample]])

            for t_length in itertools.count():

                # Epsilon for this time step
                epsilon = epsilons[min(i_episode, epsilon_decay_steps-1)]
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, current_order =  Env_calculate_transition_prob(i_sample, state, action, ordinal_error, calculate_error,
                    value_ha, value_hb, value_hdh, value_hdv, value_va, value_vb, value_vdh, value_vdv, present_order, current_order, is_hv)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]                
                td_delta = td_target - Q[state][action]
                #print(td_delta)
                Q[state][action] += alpha * td_delta
                
                if done or t_length >= 20:
                    stats.episode_lengths[i_episode] += t_length + 1
                    break

                state = next_state
            
        stats.episode_maxrewards[i_episode] = np.sum(max_EV)
        #stats.episode_accuracy[i_episode], stats.episode_test_rewards[i_episode], stats.episode_choice_calculate[i_episode], stats.episode_rational_choice[i_episode] = test_accuracy()        
        if i_episode >=converge_num:
            stats.episode_accuracy[i_episode-converge_num], stats.episode_test_rewards[i_episode-converge_num], stats.episode_choice_calculate[i_episode-converge_num], stats.episode_rational_choice[i_episode-converge_num], stats.episode_choice_poT[i_episode-converge_num], stats.episode_choice_poC[i_episode-converge_num] = test_accuracy(ordinal_error,calculate_error)

    return stats   
                                
Q = defaultdict(lambda: np.zeros(num_action))

stats = q_learning(num_episodes = 6000)

print("r0 d0.9 e0.1 a0.1 allrondom")
print(len(Q))
print("o = 0.3, c = 0.5")    
plotting.plot_episode_stats(stats)

#print(Q)


