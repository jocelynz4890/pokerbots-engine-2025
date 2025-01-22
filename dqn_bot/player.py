'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import pickle
from estimators import MonteCarloEstimator
BOUNTY_CONSTANT, BOUNTY_RATIO = 10, 1.5

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# from table import q_table, prev_round_num


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# buffer for holding recent transitions
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size): 
        """Selects random batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):

    def __init__(self, n_input, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_input, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    

BATCH_SIZE = 128 # num transitions to sample from replay buffer
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # epsilon stuff
EPS_END = 0.05
EPS_DECAY = 1000 # higher = slower decay
TAU = 0.005 # update rate of target network
LR = 1e-4 # learning rate of the ``AdamW`` optimizer

n_actions = 403 # call, check, fold, raise with values 1 to 400 (change this to be relative to max/min raise?)
state = [] 
n_input = len(state)

policy_net = DQN(n_input, n_actions).to(device)
target_net = DQN(n_input, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)



def select_action(state, legal_actions, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold: # exploit
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else: # explore
        return torch.tensor([legal_actions], device=device, dtype=torch.long)


deltas = []


def plot_deltas(show_result=False):
    plt.figure(1)
    deltas_t = torch.tensor(deltas, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('delta')
    plt.ylabel('Duration')
    plt.plot(deltas_t.numpy())
    # Take 100 delta averages and plot them too
    if len(deltas_t) >= 100:
        means = deltas_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
            
            

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            deltas.append(my_delta)
            plot_deltas()
            break

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.alpha = 0.1  # keep learning rate low for poker?
        self.gamma = 0.9  # idk
        self.epsilon = 0.95 # ! EDIT FOR SCRIM
        self.min_epsilon = 0.7 # ! EDIT FOR SCRIM
        self.decay_rate = 0.99
        self.last_action = None
        self.last_state = None

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        #my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[active]  # your cards
        #big_blind = bool(active)  # True if you are the big blind
        #my_bounty = round_state.bounties[active]  # your current bounty rank
        self.estimator = MonteCarloEstimator()
        self.last_state = None
        self.last_action = None

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        #street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[active]  # your cards
        #opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        my_bounty_hit = terminal_state.bounty_hits[active]  # True if you hit bounty
        opponent_bounty_hit = terminal_state.bounty_hits[1-active] # True if opponent hit bounty
        bounty_rank = previous_state.bounties[active]  # your bounty rank

        # The following is a demonstration of accessing illegal information (will not work)
        opponent_bounty_rank = previous_state.bounties[1-active]  # attempting to grab opponent's bounty rank

        # if my_bounty_hit:
        #     print("I hit my bounty of " + bounty_rank + "!") # ! REMOVE FOR SCRIM
        # if opponent_bounty_hit:
        #     print("Opponent hit their bounty of " + opponent_bounty_rank + "!") # ! REMOVE FOR SCRIM
            
        # learn from round, reward is delta
        self.learn(my_delta)
        
        round_num = game_state.round_num
        
        if round_num == 1000:
            plot_deltas(show_result=True)
            plt.ioff()
            plt.show()

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        
        MAX_RAISE_RATIO = 0.5 # proportion of EV to raise by
        ALL_IN_EQUITY_THRESHOLD = 0.70
        ALL_IN_PROB = 0.99
        APPROX_MAX_PREFLOP_PAYOUT = 5
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        # print("board:", round_state.deck, "hand:", my_cards) # ! REMOVE FOR SCRIM
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        rounds_left = 1001 - game_state.round_num
        bankroll = game_state.bankroll
        # if bankroll > APPROX_MAX_PREFLOP_PAYOUT * rounds_left:
        #     return FoldAction()

        equity, bounty_prob = self.estimator.estimate(my_cards, board_cards, my_bounty)
        ev = int((opp_pip + my_contribution) * (equity - bounty_prob) + ((opp_contribution) * BOUNTY_RATIO + BOUNTY_CONSTANT + my_contribution) * (bounty_prob)) # ev of payout assuming you've lost your pips
        max_wanted_raise = ev * MAX_RAISE_RATIO
        
        rankToInt = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suitToInt = {'s':0, 'h':1, 'd':2, 'c':3}
        def cardToInd(c):
            return rankToInt[c[0]] + 13 * suitToInt[c[1]]
        one_hot = zeros = [0]*52
        indices = [cardToInd(card) for card in board_cards]
        for i in indices:
            zeros[i] = 1
        state = [equity, bounty_prob, street, my_contribution, opp_contribution] + one_hot
        
        
        legal_actions_list = []
        
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            possible_raises = [min_raise, (max_raise - min_raise) // 2, max_raise] # range(min_raise, max_raise + 1, (max_raise - min_raise) // 3 or 1)
            legal_actions_list.extend(RaiseAction(amount) for amount in possible_raises)
            
        if CheckAction in legal_actions:
            legal_actions_list.append(CheckAction())
            
        if FoldAction in legal_actions:
            legal_actions_list.append(FoldAction())
            
        if CallAction in legal_actions:
            legal_actions_list.append(CallAction())
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

        action = select_action(state)
        
        return action
    
    
    def select_best_action(self, state, legal_actions):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in legal_actions}
            # print("not found")

        # this is the bottleneck
        best_action = max(legal_actions, key=lambda action: self.q_table[state].get(action, 0))
        return best_action

    def learn(self, reward):
        if self.last_state is None or self.last_action is None:
            return

        if self.last_state not in self.q_table:
            self.q_table[self.last_state] = {}

        if self.last_action not in self.q_table[self.last_state]:
            self.q_table[self.last_state][self.last_action] = 0

        prev_q_value = self.q_table[self.last_state][self.last_action]
        next_max_q_value = max(self.q_table.get(self.last_state, {}).values(), default=0)
        new_q_value = prev_q_value + self.alpha * (reward + self.gamma * next_max_q_value - prev_q_value)
        self.q_table[self.last_state][self.last_action] = new_q_value


if __name__ == '__main__':
    run_bot(Player(), parse_args())
