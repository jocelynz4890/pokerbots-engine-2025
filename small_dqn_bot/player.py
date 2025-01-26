from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import random
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
import pickle

def save_replay_memory(replay_memory, file_path="replay_memory.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(replay_memory, f)
    print("replay memory saved")
    
def load_replay_memory(replay_memory, file_path="replay_memory.pkl"):
    try:
        with open(file_path, "rb") as f:
            replay_memory = pickle.load(f)
        print("replay memory loaded")
    except FileNotFoundError:
        print("replay memory failed to load")

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

rankToInt = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
suitToInt = {'s':0, 'h':1, 'd':2, 'c':3}
def cardToInd(c):
    return rankToInt[c[0]] + 13 * suitToInt[c[1]]


# buffer for holding recent transitions
# (experience replay helps stabilize learning, prevents overftting to recent events)
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
        self.layer1 = nn.Linear(n_input, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
def save_model(policy_net, target_net, optimizer, replay_memory, file_path="model.pth"):
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    save_replay_memory(replay_memory)
    print("model parameters saved")

def load_model(policy_net, target_net, optimizer, replay_memory, file_path="model.pth"):
    try:
        params = torch.load(file_path, map_location=device)
        policy_net.load_state_dict(params['policy_net_state_dict'])
        target_net.load_state_dict(params['target_net_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        load_replay_memory(replay_memory)
        print("policy, target network, and optimizer loaded successfully")
    except FileNotFoundError:
        print("policy, target network, and optimizer load failed")
    
#######################################################################################
##    DQN globals
#######################################################################################
BATCH_SIZE = 128 # num transitions to sample from replay buffer
GAMMA = 0.99 # discount factor
EPS_START = 0.5 # epsilon stuff, was 0.9
EPS_END = 0.2
EPS_DECAY = 1000 # higher = slower decay
TAU = 0.005 # update rate of target network, was 0.005
LR = 1e-4 # learning rate of the ``AdamW`` optimizer, was 1e-4
raise_increments = [0., 0.01, 0.02, 0.2, 0.6, 1.0] # percentage of [min raise, max raise]
n_raise_increments = len(raise_increments)
n_actions = 3 + n_raise_increments # call, check, fold, TODO raise with values relative to max/min raise
state = [] 
next_state = None
n_input = 5 # len(state), 5 + 52 + 52
rewards = []
reward = None
deltas = []
action = None

# the following need to be loaded/saved
policy_net = DQN(n_input, n_actions).to(device)
target_net = DQN(n_input, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000) #torch.save doesn't save this so it has to be pickled/unpickled
load_model(policy_net, target_net, optimizer, memory)
#######################################################################################
#######################################################################################


def select_action(state, legal_actions, steps_done):
    sample = random.random()
    global n_actions
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if state is not None and sample > eps_threshold: # exploit
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = policy_net(state).max(1).indices.view(1, 1)
            print(f"model says {selected_action}")
            return selected_action
    else: # explore
        choice = None
        while choice is None or legal_actions[choice] == 0:
            choice = random.randint(0, n_actions-1)#torch.tensor([legal_actions], device=device, dtype=torch.long) # make it choose an action
        return choice
# todo: this is entirely copy pasted from plot_duration()
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
    global memory, policy_net, target_net, optimizer
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
    non_final_next_states = torch.cat([torch.tensor(s, device = device, dtype = torch.float) for s in batch.next_state])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).reshape((BATCH_SIZE, 1))
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(f"action_batch {action_batch.shape}")
    # print(f"policy net output {policy_net(state_batch).shape}")
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
        # self.alpha = 0.1  
        # self.gamma = 0.9  
        # self.epsilon = 0.95 
        # self.min_epsilon = 0.7 
        # self.decay_rate = 0.99

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
        global state, next_state, reward
        state = None
        next_state = None
        reward = 0.

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

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!") 
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!") 
        global deltas, rewards, reward, target_net, policy_net, optimizer, memory, state, action
        # deltas.append(my_delta)
        # rewards.append(my_delta)
        # plot_deltas()
        reward = my_delta

        my_contribution = STARTING_STACK - previous_state.stacks[active]
        opp_contribution = STARTING_STACK - previous_state.stacks[1-active]
        board_cards = previous_state.deck[:5]
        my_cards = previous_state.hands[active]
        board_card_encoding = [0]*52
        indices = [cardToInd(card) for card in (board_cards)]
        for i in indices:
            board_card_encoding[i] = 1
        my_card_encoding = [0]*52
        indices = [cardToInd(card) for card in my_cards]
        for i in indices:
            my_card_encoding[i] = 1

        #train on terminal state

        next_state = [my_delta >0, my_bounty_hit, 5, my_contribution, opp_contribution]# + board_card_encoding + my_card_encoding
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        if state is not None:
            memory.push(state, action, next_state, torch.tensor([my_delta], device = device))
            # print(f"pushing next_state, action, reward {(next_state, action, my_delta)}")
                
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        # target net gradually incorporates policy net's weights while retaining some of its own previous weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # handle game over
        # if game_state.round_num == 1000:
        #     save_model(policy_net, target_net, optimizer, memory)
        #     plot_deltas(show_result=True)
        #     plt.ioff()
        #     plt.show()
        #     print(f"deltas {deltas}")

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
        #######################################################################################
        ##    CALCULATIONS, FOLDING, AND EQUITY
        #######################################################################################
        MAX_RAISE_RATIO = 0.5 # proportion of EV to raise by
        ALL_IN_EQUITY_THRESHOLD = 0.70
        ALL_IN_PROB = 0.99
        APPROX_MAX_PREFLOP_PAYOUT = 5
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        print("board:", round_state.deck, "hand:", my_cards) # ! REMOVE FOR SCRIM
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        # rounds_left = 1001 - game_state.round_num
        # bankroll = game_state.bankroll
        # if bankroll > APPROX_MAX_PREFLOP_PAYOUT * rounds_left:
        #     return FoldAction()        

        equity, bounty_prob = self.estimator.estimate(my_cards, board_cards, my_bounty)
        ev = int((opp_contribution + my_contribution) * (equity - bounty_prob) + ((opp_contribution) * BOUNTY_RATIO + BOUNTY_CONSTANT + my_contribution) * (bounty_prob)) # ev of payout assuming you've lost your pips
        max_wanted_raise = ev * MAX_RAISE_RATIO
        #######################################################################################
        #######################################################################################



        
        #######################################################################################
        ##    TRAINING LOOP
        #######################################################################################
        
        # get the current state info
        global state, memory, target_net, policy_net, action, n_raise_increments

        #board_card_encoding = [0]*52
        #indices = [cardToInd(card) for card in (board_cards)]
        #for i in indices:
        #    board_card_encoding[i] = 1
        #my_card_encoding = [0]*52
        #indices = [cardToInd(card) for card in my_cards]
        #for i in indices:
        #    my_card_encoding[i] = 1
        # Move to the "next" state (by entering the get_action method again, our current state = next state from last time)
        next_state = [equity, bounty_prob, street, my_contribution, opp_contribution]# + board_card_encoding + my_card_encoding
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        if state is not None: # will only be false during pre-flop, when state is set to []
            
        
            # now we will: finish processing the "next" state:
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store the transition in memory
            memory.push(state, action, next_state, torch.tensor([reward], device = device))
            # print(f"pushing next_state, action, reward (non-terminal) {(next_state, action, reward)}")
            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # target net gradually incorporates policy net's weights while retaining some of its own previous weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
            # now we can say that our current state is what was previously the next state
        state = next_state
        
        # make new list of legal actions (with raise amounts specified)
        # also verified that equality checks work on them, like RaiseAction(amount=5) == RaiseAction(amount=5) is true
        min_raise, max_raise = round_state.raise_bounds()
        model_actions_list = [CallAction(), CheckAction(), FoldAction()] + \
             [RaiseAction(int(min_raise + p*(max_raise - min_raise))) for p in raise_increments]
        legal_actions_list = []
        if CheckAction in legal_actions:
            legal_actions_list.append(CheckAction())
        else:
            legal_actions_list.append(0)
        if FoldAction in legal_actions:
            legal_actions_list.append(FoldAction())
        else:
            legal_actions_list.append(0)
        if CallAction in legal_actions:
            legal_actions_list.append(CallAction())
        else:
            legal_actions_list.append(0)
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            possible_raises = [int(min_raise + p*(max_raise - min_raise)) for p in raise_increments] # TODO
            legal_actions_list.extend(RaiseAction(amount) for amount in possible_raises)
        else:
            legal_actions_list += [0] * n_raise_increments
            
        
            

        # get action
        action = torch.tensor([select_action(state, legal_actions_list, game_state.round_num)], device = device)
        
        # reward is concerning because delta is only calculated after an entire round (after river)
        # so for preflop, flop, and turn we need some sort of intermediate reward. maybe use change in equity
        rewards = 0. 
        rewards = torch.tensor([rewards], device=device) 
        
        #######################################################################################
        #######################################################################################
        
        return model_actions_list[action]


if __name__ == '__main__':
    run_bot(Player(), parse_args())
