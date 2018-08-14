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
from lib import plotting
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

num_action = 6
#num_state_elements = 6    #num_state_elements = 6
#num_state = 4**num_state_elements     #num_state = 100
#value_state = defaultdict(lambda: np.zeros(num_state_elements))
space_action = range(num_action)

train_size = 1
num_size = 1000
window_num = 100
guass_mu, guass_sigma = 0, 2 #   3, 2; 100, 5mean and standard deviation
#beta_a, beta_b=  2, 2
beta_a, beta_b=  1, 1
location = 19.60
scale = 8.08
df = 100

def Env_calculate_transition_prob(i_sample, current_s, action,
    value_v_a, value_v_b, value_v_d, value_p_a, value_p_b, value_p_d):
    tolerance_p = 0.011
    #tolerance_v = 8.8
    tolerance_v = 1.1
    #tolerance_v = 0.1
    value_new_state = np.array(current_s)
    value_e_a = value_p_a * value_v_a
    value_e_b = value_p_b * value_v_b
    value_e_d = value_p_d * value_v_d

    calculate_noise = np.random.normal(guass_mu, guass_sigma, 1)
    '''
    if action == space_action[0]:
        if (value_p_a[i_sample] > value_p_b[i_sample] + tolerance_p):
            value_new_state[0] = 1
        elif (value_p_a[i_sample] < value_p_b[i_sample] + tolerance_p) and (value_p_a[i_sample] > value_p_b[i_sample] - tolerance_p):
            value_new_state[0] = 2
        else:
            value_new_state[0] = 3
    elif action == space_action[1]:
        if (value_p_a[i_sample] > value_p_d[i_sample] + tolerance_p):
            value_new_state[1] = 1
        elif (value_p_a[i_sample] < value_p_d[i_sample] + tolerance_p) and (value_p_a[i_sample] > value_p_d[i_sample] - tolerance_p):
            value_new_state[1] = 2
        else:
            value_new_state[1] = 3
    elif action == space_action[2]:
        if (value_p_b[i_sample] > value_p_d[i_sample] + tolerance_p):
            value_new_state[2] = 1
        elif (value_p_b[i_sample] < value_p_d[i_sample] + tolerance_p) and (value_p_b[i_sample] > value_p_d[i_sample] - tolerance_p):
            value_new_state[2] = 2
        else:
            value_new_state[2] = 3
    elif action == space_action[3]:
        if (value_v_a[i_sample] > value_v_b[i_sample] + tolerance_v):
            value_new_state[3] = 1
        elif (value_v_a[i_sample] < value_v_b[i_sample] + tolerance_v) and (value_v_a[i_sample] > value_v_b[i_sample] - tolerance_v):
            value_new_state[3] = 2
        else:
            value_new_state[3] = 3
    elif action == space_action[4]:
        if (value_v_a[i_sample] > value_v_d[i_sample] + tolerance_v):
            value_new_state[4] = 1
        elif (value_v_a[i_sample] < value_v_d[i_sample] + tolerance_v) and (value_v_a[i_sample] > value_v_d[i_sample] - tolerance_v):
            value_new_state[4] = 2
        else:
            value_new_state[4] = 3
    elif action == space_action[5]:
        if (value_v_b[i_sample] > value_v_d[i_sample] + tolerance_v):
            value_new_state[5] = 1
        elif (value_v_b[i_sample] < value_v_d[i_sample] + tolerance_v) and (value_v_b[i_sample] > value_v_d[i_sample] - tolerance_v):
            value_new_state[5] = 2
        else:
            value_new_state[5] = 3
    '''
    if action == space_action[0]:
        value_new_state[0] = round(value_p_a[i_sample] * value_v_a[i_sample] + calculate_noise)
    elif action == space_action[1]:
        value_new_state[1] = round(value_p_b[i_sample] * value_v_b[i_sample] + calculate_noise)
    elif action == space_action[2]:
        value_new_state[2] = round(value_p_d[i_sample] * value_v_d[i_sample] + calculate_noise)
    else:
        value_new_state = np.array(current_s)



    #new_state =  int(value_new_state[0]*4**5 + value_new_state[1]*4**4 +
    #              value_new_state[2]*4**3 + value_new_state[3]*4**2 +
    #              value_new_state[4]*4**1 + value_new_state[5])
    
    maximum_ev = np.max([value_e_a[i_sample], value_e_b[i_sample], value_e_d[i_sample]])
    if action == space_action[3]:
        is_done = True
        reward = 10 if value_e_a[i_sample] == maximum_ev else -10
        #reward = value_e_a[i_sample]
        #reward = np.random.choice([value_v_a[i_sample], 0], p=[value_p_a[i_sample], 1 - value_p_a[i_sample]])
    elif action == space_action[4]:
        is_done = True
        reward = 10 if value_e_b[i_sample] == maximum_ev else -10
        #reward = value_e_b[i_sample]
        #reward = np.random.choice([value_v_b[i_sample], 0], p=[value_p_b[i_sample], 1 - value_p_b[i_sample]])
    elif action == space_action[5]:
        is_done = True
        reward = 10 if value_e_d[i_sample] == maximum_ev else -10
        #reward = value_e_d[i_sample]
        #reward = np.random.choice([value_v_d[i_sample], 0], p=[value_p_d[i_sample], 1 - value_p_d[i_sample]])
    else:
        is_done = False
        reward = -1
    
    #reward = 100.0 if (new_state == num_state - 1) else -1.0
    return tuple(value_new_state), reward, is_done


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


def GenerateData():
    '''
    from scipy.stats import t
    test_v_a = t.rvs(df, location, scale, 1)
    test_v_b = t.rvs(df, location, scale, 1)
    test_v_d = t.rvs(df, location, scale, 1)     
    test_p_a = np.random.beta(beta_a, beta_b, 1)
    test_p_b = np.random.beta(beta_a, beta_b, 1)
    test_p_d = np.random.beta(beta_a, beta_b, 1)
    return round(test_v_a[0], 2), round(test_v_b[0], 2), round(test_v_d[0], 2), round(test_p_a[0], 2), round(test_p_b[0], 2), round(test_p_d[0],2)
    '''
    #'''
    is_mached = False
    from scipy.stats import t
    while is_mached !=True:
        test_v_a = t.rvs(df, location, scale, 1)
        test_v_b = t.rvs(df, location, scale, 1)
        test_v_d = t.rvs(df, location, scale, 1)
        #test_v_a = np.random.normal(guass_mu, guass_sigma, 1)
        #test_v_b = np.random.normal(guass_mu, guass_sigma, 1)
        #test_v_d = np.random.normal(guass_mu, guass_sigma, 1)       
        test_p_a = np.random.beta(beta_a, beta_b, 1)
        test_p_b = np.random.beta(beta_a, beta_b, 1)
        test_p_d = np.random.beta(beta_a, beta_b, 1)
        test_e_a = test_p_a * test_v_a
        test_e_b = test_p_b * test_v_b
        test_e_d = test_p_d * test_v_d
        #return test_v_a[0], test_v_b[0], test_v_d[0], test_p_a[0], test_p_b[0], test_p_d[0]
        #return round(test_v_a[0], 2), round(test_v_b[0], 2), round(test_v_d[0], 2), round(test_p_a[0], 2), round(test_p_b[0], 2), round(test_p_d[0],2)
        
        if (test_p_a[0] > test_p_b[0]) and (test_p_b[0] > test_p_d[0]) and (test_v_b[0] > test_v_d[0]) and (test_v_d[0] > test_v_a[0]):
        #if (test_p_a[0] > test_p_d[0]) and (test_p_d[0] > test_p_b[0]) and (test_v_b[0] > test_v_a[0]) and (test_v_a[0] > test_v_d[0]):
        #if (test_p_a[0] > test_p_d[0]) and (test_p_d[0] > test_p_b[0]) and (test_v_b[0] > test_v_a[0]) and (test_v_a[0] > test_v_d[0]) and (abs(test_e_a - test_e_b) < 0.1):
        #if (test_p_a[0] > test_p_b[0]) and (test_p_b[0] > test_p_d[0]) and (test_v_b[0] > test_v_d[0]) and (test_v_d[0] > test_v_a[0]):
        #if (test_p_a[0] > test_p_b[0]) and (test_p_b[0] > test_p_d[0]) and (test_v_b[0] > test_v_a[0]) and (test_v_a[0] > test_v_d[0]):
            is_mached = True
            return round(test_v_a[0], 2), round(test_v_b[0], 2), round(test_v_d[0], 2), round(test_p_a[0], 2), round(test_p_b[0], 2), round(test_p_d[0],2)
            #return test_v_a[0], test_v_b[0], test_v_d[0], test_p_a[0], test_p_b[0], test_p_d[0]
        else:
            is_mached = False
    #'''

def test_accuracy():
    test_size = 100
    test_v_a = np.zeros(test_size)
    test_v_b = np.zeros(test_size)
    test_v_d = np.zeros(test_size)
    test_p_a = np.zeros(test_size)
    test_p_b = np.zeros(test_size)
    test_p_d = np.zeros(test_size)
    for i_test in range(test_size):
        test_v_a[i_test], test_v_b[i_test], test_v_d[i_test], test_p_a[i_test], test_p_b[i_test], test_p_d[i_test] = GenerateData()
    
    test_e_a = test_p_a * test_v_a
    test_e_b = test_p_b * test_v_b
    test_e_d = test_p_d * test_v_d

    choice_calculate = np.zeros(3)
    choice_true = 0
    calculative_reward = 0
    rational_choice = np.zeros(3)
    #test_size = len(test_v_a)
    for i_test in range(test_size):
        state = (0, 0, 0)
        done = 0
        action = 0       
        for t in itertools.count():

            policy = make_epsilon_greedy_policy(Q, num_action) 
            action_probs = policy(state, 0.0)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done =  Env_calculate_transition_prob(i_test, state, action,
            test_v_a, test_v_b, test_v_d, test_p_a, test_p_b, test_p_d)
            calculative_reward += reward

            if done or t == 10:
                break
        
            state = next_state
        
        best_action = action
        if best_action == 3:
            choice_calculate[0] += 1
        elif best_action == 4:
            choice_calculate[1] += 1
        elif best_action == 5:
            choice_calculate[2] += 1

        max_EV = np.argmax([test_e_a[i_test], test_e_b[i_test], test_e_d[i_test]])
        if max_EV == 0:
            rational_choice[0] += 1
        elif max_EV == 1:
            rational_choice[1] += 1
        elif max_EV == 2:
            rational_choice[2] += 1

        if best_action == (max_EV + 3): choice_true += 1

    choice_accuracy = choice_true / test_size
    choice_calculate = choice_calculate / test_size
    rational_choice = rational_choice / test_size

    return choice_accuracy, calculative_reward, choice_calculate, rational_choice


def q_learning(num_episodes, discount_factor=0.9, alpha=0.1, 
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
        episode_rational_choice=np.zeros((cal_num, 3)))
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(Q, num_action)
    epsilon = epsilon_start

    for i_episode in range(num_episodes):
        from scipy.stats import t
        value_v_a = t.rvs(df, location, scale, num_size)
        value_v_b = t.rvs(df, location, scale, num_size)
        value_v_d = t.rvs(df, location, scale, num_size) 
        value_p_a = np.random.beta(beta_a, beta_b,num_size)
        value_p_b = np.random.beta(beta_a, beta_b,num_size)
        value_p_d = np.random.beta(beta_a, beta_b,num_size)
        value_v_a = np.round(value_v_a, 2)
        value_v_b = np.round(value_v_b, 2)
        value_v_d = np.round(value_v_d, 2)
        value_p_a = np.round(value_p_a, 2)
        value_p_b = np.round(value_p_b, 2)
        value_p_d = np.round(value_p_d, 2)
        '''
        value_v_a[:(num_size-window_num)] = [v for v in value_v_a[(window_num-num_size):]]
        value_v_a[-window_num:] = np.round(np.random.normal(guass_mu, guass_sigma, window_num), 2)
        value_v_b[:(num_size-window_num)] = [v for v in value_v_b[(window_num-num_size):]]
        value_v_b[-window_num:] = np.round(np.random.normal(guass_mu, guass_sigma, window_num), 2)
        value_v_d[:(num_size-window_num)] = [v for v in value_v_d[(window_num-num_size):]]
        value_v_d[-window_num:] = np.round(np.random.normal(guass_mu, guass_sigma, window_num), 2)
        value_p_a[:(num_size-window_num)] = [v for v in value_p_a[(window_num-num_size):]]
        value_p_a[-window_num:] = np.round(np.random.beta(beta_a, beta_b, window_num), 2)
        value_p_b[:(num_size-window_num)] = [v for v in value_p_b[(window_num-num_size):]]
        value_p_b[-window_num:] = np.round(np.random.beta(beta_a, beta_b, window_num), 2)
        value_p_d[:(num_size-window_num)] = [v for v in value_p_d[(window_num-num_size):]]
        value_p_d[-window_num:] = np.round(np.random.beta(beta_a, beta_b, window_num), 2)         
        '''
        value_e_a = value_p_a * value_v_a
        value_e_b = value_p_b * value_v_b
        value_e_d = value_p_d * value_v_d
        max_EV = np.zeros(num_size)
            
        if (i_episode + 1) % 100 == 0:
            print("\rEpisodes {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush() 
        t_length = 0

        for i_sample in range(num_size):

            state = (-100, -100, -100)
            max_EV[i_sample] = np.max([value_e_a[i_sample], value_e_b[i_sample], value_e_d[i_sample]])

            for t_length in itertools.count():

                # Epsilon for this time step
                epsilon = epsilons[min(i_episode, epsilon_decay_steps-1)]
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done =  Env_calculate_transition_prob(i_sample, state, action,
                    value_v_a, value_v_b, value_v_d, value_p_a, value_p_b, value_p_d)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]                
                td_delta = td_target - Q[state][action]
                #print(td_delta)
                Q[state][action] += alpha * td_delta
                
                if done or t_length == 13:
                    stats.episode_lengths[i_episode] += t_length + 1
                    break

                state = next_state
            
        stats.episode_maxrewards[i_episode] = np.sum(max_EV)
        #stats.episode_accuracy[i_episode], stats.episode_test_rewards[i_episode], stats.episode_choice_calculate[i_episode], stats.episode_rational_choice[i_episode] = test_accuracy()        
        if i_episode >=converge_num:
            stats.episode_accuracy[i_episode-converge_num], stats.episode_test_rewards[i_episode-converge_num], stats.episode_choice_calculate[i_episode-converge_num], stats.episode_rational_choice[i_episode-converge_num] = test_accuracy()

    return stats   
                                

Q = defaultdict(lambda: np.zeros(num_action))

stats = q_learning(num_episodes = 10000)

print("r0 d0.9 e0.1 a0.1 allrondom")
print(len(Q))    
plotting.plot_episode_stats(stats)

print(Q)
