'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
from estimators import MonteCarloEstimator
BOUNTY_CONSTANT, BOUNTY_RATIO = 10, 1.5

from table import q_table, prev_round_num

import os

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
        self.q_table = q_table  
        self.alpha = 0.1  # keep learning rate low for poker?
        self.gamma = 0.9  # idk
        self.epsilon = 0.95
        self.min_epsilon = 0.1
        self.decay_rate = 0.99
        self.last_action = None
        self.last_state = None
        # print(RaiseAction(amount=10) == RaiseAction(amount=10)) made sure comparison works on actions

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

        if my_bounty_hit:
            print("I hit my bounty of " + bounty_rank + "!")
        if opponent_bounty_hit:
            print("Opponent hit their bounty of " + opponent_bounty_rank + "!")
            
        # learn from round, reward is delta
        self.learn(my_delta)
        
        round_num = game_state.round_num
        
        if round_num == 1000:
            # print(self.q_table)

            # had to gpt this lmao
            # ! REMOVE WHEN USING IN SCRIM
            # Assuming self.q_table is the data you want to write
            file_path = os.path.join(os.getcwd(), 'table.py')

            # Open the file in write mode
            with open(file_path, 'w') as file:
                file.write(f"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

prev_round_num = {round_num + prev_round_num}
                            
q_table = {self.q_table}
                            """)

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
        print("board:", round_state.deck, "hand:", my_cards)
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
        
        # state = (ev, my_pip, my_stack, my_contribution, street, opp_pip, opp_stack, opp_contribution)
        # state = (ev, street, my_pip, opp_pip, )
        state = (ev, street, my_pip, my_contribution, opp_pip, opp_contribution)
        # state = (ev, street) too simple, trained ~3 mil rounds and it got stuck at -16000 for 1000 round games
        
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

        if random.random() < self.epsilon:
            action = random.choice(legal_actions_list)
        else:
            action = self.select_best_action(state, legal_actions_list)

        self.last_state = state
        self.last_action = action
        
        return action
    
    
    def select_best_action(self, state, legal_actions):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in legal_actions}

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
