'''
setup.py
'''
from setuptools import setup
from Cython.Build import cythonize


# compile using 'python setup.py build_ext --inplace'




setup(
   ext_modules=cythonize("estimators_cython.pyx")
)


setup(
   ext_modules=cythonize("player_cython.pyx")
)








'''
estimator_cython.pyx
'''
import eval7
import numpy
from typing import List, Tuple

class EquityEstimator:
    def estimate(self, my_cards: List[str], board_cards: List[str], my_bounty: str) -> Tuple[float, float]:
        pass
    
class MonteCarloEstimator(EquityEstimator):
    def estimate(self, my_cards: List[str], board_cards: List[str], my_bounty: str) -> Tuple[float, float]:
        hand_strength: dict[str, int] = {'AAo': 85, 'AAs': 85, 'AKs': 67, 'AQs': 66, 'AJs': 65, 'ATs': 65, 'A9s': 63, 'A8s': 62, 'A7s': 61, 'A6s': 60, 'A5s': 60, 'A4s': 59, 'A3s': 58, 'A2s': 57, 'AKo': 65, 'KKo': 82, 'KKs': 82, 'KQs': 63, 'KJs': 63, 'KTs': 62, 'K9s': 60, 'K8s': 58, 'K7s': 57, 'K6s': 57, 'K5s': 56, 'K4s': 55, 'K3s': 54, 'K2s': 53, 'AQo': 64, 'KQo': 61, 'QQo': 80, 'QQs': 80, 'QJs': 60, 'QTs': 59, 'Q9s': 58, 'Q8s': 56, 'Q7s': 54, 'Q6s': 54, 'Q5s': 53, 'Q4s': 52, 'Q3s': 51, 'Q2s': 50, 'AJo': 64, 'KJo': 61, 'QJo': 60, 'JJo': 77, 'JJs': 77, 'JTs': 57, 'J9s': 56, 'J8s': 54, 'J7s': 52, 'J6s': 50, 'J5s': 50, 'J4s': 49, 'J3s': 48, 'J2s': 47, 'ATo': 63, 'KTo': 60, 'QTo': 57, 'JTo': 55, 'TTo': 75, 'TTs': 75, 'T9s': 54, 'T8s': 52, 'T7s': 51, 'T6s': 49, 'T5s': 47, 'T4s': 46, 'T3s': 46, 'T2s': 45, 'A9o': 61, 'K9o': 58, 'Q9o': 55, 'J9o': 53, 'T9o': 51, '99o': 72, '99s': 72, '98s': 51, '97s': 49, '96s': 47, '95s': 46, '94s': 44, '93s': 43, '92s': 42, 'A8o': 60, 'K8o': 56, 'Q8o': 54, 'J8o': 51, 'T8o': 50, '98o': 48, '88o': 69, '88s': 69, '87s': 48, '86s': 46, '85s': 44, '84s': 43, '83s': 41, '82s': 40, 'A7o': 59, 'K7o': 55, 'Q7o': 52, 'J7o': 50, 'T7o': 48, '97o': 46, '87o': 45, '77o': 66, '77s': 66, '76s': 45, '75s': 44, '74s': 42, '73s': 40, '72s': 38, 'A6o': 58, 'K6o': 54, 'Q6o': 51, 'J6o': 48, 'T6o': 46, '96o': 44, '86o': 43, '76o': 42, '66o': 63, '66s': 63, '65s': 43, '64s': 41, '63s': 39, '62s': 38, 'A5o': 58, 'K5o': 53, 'Q5o': 50, 'J5o': 47, 'T5o': 44, '95o': 43, '85o': 41, '75o': 40, '65o': 40, '55o': 60, '55s': 60, '54s': 41, '53s': 40, '52s': 38, 'A4o': 57, 'K4o': 52, 'Q4o': 49, 'J4o': 46, 'T4o': 43, '94o': 41, '84o': 39, '74o': 38, '64o': 40, '54o': 38, '44o': 57, '44s': 57, '43s': 39, '42s': 37, 'A3o': 56, 'K3o': 51, 'Q3o': 48, 'J3o': 45, 'T3o': 42, '93o': 40, '83o': 37, '73o': 37, '63o': 36, '53o': 36, '43o': 35, '33o': 54, '33s': 54, '32s': 36, 'A2o': 55, 'K2o': 50, 'Q2o': 47, 'J2o': 44, 'T2o': 42, '92o': 39, '82o': 37, '72o': 35, '62o': 34, '52o': 34, '42o': 33, '32o': 32, '22o': 50, '22s': 50, 'KAs': 67, 'QAs': 66, 'JAs': 65, 'TAs': 65, '9As': 63, '8As': 62, '7As': 61, '6As': 60, '5As': 60, '4As': 59, '3As': 58, '2As': 57, 'KAo': 65, 'QKs': 63, 'JKs': 63, 'TKs': 62, '9Ks': 60, '8Ks': 58, '7Ks': 57, '6Ks': 57, '5Ks': 56, '4Ks': 55, '3Ks': 54, '2Ks': 53, 'QAo': 64, 'QKo': 61, 'JQs': 60, 'TQs': 59, '9Qs': 58, '8Qs': 56, '7Qs': 54, '6Qs': 54, '5Qs': 53, '4Qs': 52, '3Qs': 51, '2Qs': 50, 'JAo': 64, 'JKo': 61, 'JQo': 60, 'TJs': 57, '9Js': 56, '8Js': 54, '7Js': 52, '6Js': 50, '5Js': 50, '4Js': 49, '3Js': 48, '2Js': 47, 'TAo': 63, 'TKo': 60, 'TQo': 57, 'TJo': 55, '9Ts': 54, '8Ts': 52, '7Ts': 51, '6Ts': 49, '5Ts': 47, '4Ts': 46, '3Ts': 46, '2Ts': 45, '9Ao': 61, '9Ko': 58, '9Qo': 55, '9Jo': 53, '9To': 51, '89s': 51, '79s': 49, '69s': 47, '59s': 46, '49s': 44, '39s': 43, '29s': 42, '8Ao': 60, '8Ko': 56, '8Qo': 54, '8Jo': 51, '8To': 50, '89o': 48, '78s': 48, '68s': 46, '58s': 44, '48s': 43, '38s': 41, '28s': 40, '7Ao': 59, '7Ko': 55, '7Qo': 52, '7Jo': 50, '7To': 48, '79o': 46, '78o': 45, '67s': 45, '57s': 44, '47s': 42, '37s': 40, '27s': 38, '6Ao': 58, '6Ko': 54, '6Qo': 51, '6Jo': 48, '6To': 46, '69o': 44, '68o': 43, '67o': 42, '56s': 43, '46s': 41, '36s': 39, '26s': 38, '5Ao': 58, '5Ko': 53, '5Qo': 50, '5Jo': 47, '5To': 44, '59o': 43, '58o': 41, '57o': 40, '56o': 40, '45s': 41, '35s': 40, '25s': 38, '4Ao': 57, '4Ko': 52, '4Qo': 49, '4Jo': 46, '4To': 43, '49o': 41, '48o': 39, '47o': 38, '46o': 40, '45o': 38, '34s': 39, '24s': 37, '3Ao': 56, '3Ko': 51, '3Qo': 48, '3Jo': 45, '3To': 42, '39o': 40, '38o': 37, '37o': 37, '36o': 36, '35o': 36, '34o': 35, '23s': 36, '2Ao': 55, '2Ko': 50, '2Qo': 47, '2Jo': 44, '2To': 42, '29o': 39, '28o': 37, '27o': 35, '26o': 34, '25o': 34, '24o': 33, '23o': 32}

        SIMULATION_ROUNDS: int = 1000
        ranks: Tuple[str, ...] = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
        suits: Tuple[str, ...] = ('c', 'd', 'h', 's')
        my_cards: List[eval7.Card] = list(map(eval7.Card, my_cards))
        board_cards: List[eval7.Card] = list(map(eval7.Card, board_cards))
        revealed_cards: List[eval7.Card] = my_cards + board_cards
        street: int = len(board_cards)

        if not street:
            hand_ranks: List[str] = [ranks[c.rank] for c in my_cards]
            prob: float = hand_strength["".join(hand_ranks + ["s" if suits[my_cards[0].suit] == suits[my_cards[0].suit] else "o"])] / 100
            return prob, prob if my_bounty in hand_ranks else 1 - (48 / 52) ** 5

        deck: eval7.Deck = eval7.Deck()
        deck.cards = [card for card in deck.cards if card not in revealed_cards]
        selected_cards: numpy.ndarray = numpy.random.randint(0, 52 - len(revealed_cards), size=(SIMULATION_ROUNDS, 7 - street))
        wins: int = 0
        ties: int = 0
        bounty_wins: int = 0

        for _ in range(SIMULATION_ROUNDS):
            if not len(numpy.unique(selected_cards[_])) == len(selected_cards[_]):
                ties += 1
                continue
            new_board_cards: List[eval7.Card] = board_cards + [deck[i] for i in selected_cards[_, :5 - street]]
            revealed_cards: List[eval7.Card] = my_cards + new_board_cards
            opp_cards: List[eval7.Card] = [deck[i] for i in selected_cards[_, 5 - street:]]
            my_score: int = eval7.evaluate(revealed_cards)
            opp_score: int = eval7.evaluate(opp_cards + new_board_cards)
            wins += (my_score > opp_score)
            bounty_wins += ((my_score > opp_score) and (my_bounty in map(lambda c: ranks[c.rank], revealed_cards)))
            ties += (my_score == opp_score)

        if ties == SIMULATION_ROUNDS:
            return 0, 0
        return wins / (SIMULATION_ROUNDS - ties), bounty_wins / (SIMULATION_ROUNDS - ties)



















# def state_to_int(ev, street, my_pip, my_contribution, opp_pip, opp_contribution):
#     state_int =(str(ev).rjust(3,'0')+str(street)+str(my_pip).rjust(3,'0')+str(my_contribution).rjust(3,'0')+str(opp_pip).rjust(3,'0')+str(opp_contribution).rjust(3,'0'))
#     return int(state_int)

# cython likes the types to be annotated
def state_to_int(ev: int, street: int, my_pip: int, my_contribution: int, opp_pip: int, opp_contribution: int) -> int:
    state_int = (str(ev).rjust(3, '0') + str(street) + str(my_pip).rjust(3, '0') +
                 str(my_contribution).rjust(3, '0') + str(opp_pip).rjust(3, '0') + str(opp_contribution).rjust(3, '0'))
    return int(state_int)








'''
player_cython.pyx
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import pickle
from estimators import MonteCarloEstimator
# from player_cython import state_to_int

from table import q_table, prev_round_num

import os

pickle_file_path = os.path.join(os.getcwd(), 'table.pkl')

# cython likes the types to be annotated
def state_to_int(ev: int, street: int, my_pip: int, my_contribution: int, opp_pip: int, opp_contribution: int) -> int:
    state_int = (str(ev).rjust(3, '0') + str(street) + str(my_pip).rjust(3, '0') +
                 str(my_contribution).rjust(3, '0') + str(opp_pip).rjust(3, '0') + str(opp_contribution).rjust(3, '0'))
    return int(state_int)


# def convert_old_qtable(old_qtable):
#     new_qtable = {}
    
#     for old_state, action_rewards in old_qtable.items():
#         ev, street, my_pip, my_contribution, opp_pip, opp_contribution = old_state
#         new_state = state_to_int(ev, street, my_pip, my_contribution, opp_pip, opp_contribution)
        
#         new_qtable[new_state] = {}
#         for action, reward in action_rewards.items():
#             new_qtable[new_state][action] = reward
            
#     return new_qtable

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
        if os.path.exists(pickle_file_path) and os.path.getsize(pickle_file_path) > 0:
            with open(pickle_file_path, 'rb') as file:
                self.q_table = pickle.load(file)
        else:
            self.q_table = q_table
        # self.q_table = q_table
        # print("q_table loaded from pickle") # ! REMOVE FOR SCRIM
        self.alpha = 0.1  # keep learning rate low for poker?
        self.gamma = 0.9  # idk
        self.epsilon = 0 #.95 # ! EDIT FOR SCRIM
        self.min_epsilon = 0 #.7 # ! EDIT FOR SCRIM
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

        # if my_bounty_hit:
        #     print("I hit my bounty of " + bounty_rank + "!") # ! REMOVE FOR SCRIM
        # if opponent_bounty_hit:
        #     print("Opponent hit their bounty of " + opponent_bounty_rank + "!") # ! REMOVE FOR SCRIM
            
        # learn from round, reward is delta
        self.learn(my_delta)
        
        round_num = game_state.round_num
        
        if round_num == 1000:
            # print(self.q_table)
            # ! FOR SCRIM, PRINT Q_TABLE AT THE END TO LEARN (or just state + action + reward to manually update later)

            # ! REMOVE WHEN USING IN SCRIM
            try: 
                with open(pickle_file_path, 'wb') as file:
                    pickle.dump(self.q_table, file)
                # print("q_table has been pickled")
            except: 
                file_path = os.path.join(os.getcwd(), 'table.py')

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
        
        # MAX_RAISE_RATIO = 0.5 # proportion of EV to raise by
        # ALL_IN_EQUITY_THRESHOLD = 0.70
        # ALL_IN_PROB = 0.99
        # APPROX_MAX_PREFLOP_PAYOUT = 5
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        street = round_state.street  # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        # print("board:", round_state.deck, "hand:", my_cards) # ! REMOVE FOR SCRIM
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1-active]  # the number of chips your opponent has remaining
        # continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        my_bounty = round_state.bounties[active]  # your current bounty rank
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot

        rounds_left = 1001 - game_state.round_num
        bankroll = game_state.bankroll
        # if bankroll > APPROX_MAX_PREFLOP_PAYOUT * rounds_left:
        #     return FoldAction()
        
        BOUNTY_CONSTANT, BOUNTY_RATIO = 10, 1.5

        equity, bounty_prob = self.estimator.estimate(my_cards, board_cards, my_bounty)
        ev = int((opp_pip + my_contribution) * (equity - bounty_prob) + ((opp_contribution) * BOUNTY_RATIO + BOUNTY_CONSTANT + my_contribution) * (bounty_prob)) # ev of payout assuming you've lost your pips
        # max_wanted_raise = ev * MAX_RAISE_RATIO
        
        # state = (ev, my_pip, my_stack, my_contribution, street, opp_pip, opp_stack, opp_contribution)
        # state = (ev, street, my_pip, opp_pip, )
        state = state_to_int(ev, street, my_pip, my_contribution, opp_pip, opp_contribution)
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
