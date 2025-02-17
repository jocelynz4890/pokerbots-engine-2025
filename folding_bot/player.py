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
        pass

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
        pass

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
        #my_delta = terminal_state.deltas[active]  # your bankroll change from this round
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
        PREFLOP_FOLD_EV_THRESHOLD = 1.7 # [0, 3]
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
        ev = (opp_pip + my_pip) * (equity - bounty_prob) + ((opp_pip) * BOUNTY_RATIO + BOUNTY_CONSTANT + my_pip) * (bounty_prob) # ev of payout assuming you've lost your pips in [0, pot]
        max_wanted_raise = ev * MAX_RAISE_RATIO
        print(f"round {game_state.round_num}, equity {equity}, ev {ev}")
        #preflop folding
        if ev < PREFLOP_FOLD_EV_THRESHOLD and street == 0:
            return FoldAction()
        
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()  # the smallest and largest numbers of chips for a legal bet/raise
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
            if equity > ALL_IN_EQUITY_THRESHOLD and random.random() < ALL_IN_PROB:
                return RaiseAction(max_raise)
            if max_wanted_raise > min_cost:
                return RaiseAction(int(min(max_wanted_raise, max_cost)) + my_pip)
        if ev > continue_cost and CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()
        
        
        
        # if RaiseAction in legal_actions:
        #     if estimate[0] > 0.6:
        #         return RaiseAction(max_raise)
        # if estimate[0] < 0.4:
        #     return FoldAction()
        # if CheckAction in legal_actions:  # check-call
        #     return CheckAction()
        # return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
