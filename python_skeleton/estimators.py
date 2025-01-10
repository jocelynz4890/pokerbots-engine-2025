import eval7
import numpy

class EquityEstimator:

    def estimate(self, my_cards, board_cards, my_bounty):
        pass
    
class MonteCarloEstimator(EquityEstimator):

    def estimate(self, my_cards, board_cards, my_bounty):
        SIMULATION_ROUNDS = 100
        my_cards = list(map(eval7.Card, my_cards))
        board_cards = list(map(eval7.Card, board_cards))
        revealed_cards = my_cards + board_cards
        street = len(board_cards)
        deck = eval7.Deck()
        deck.cards = [card for card in deck.cards if card not in revealed_cards]
        selected_cards = numpy.random.randint(0,52-len(revealed_cards),size=(SIMULATION_ROUNDS,7-street))
        wins = 0
        ties = 0
        bounty_wins = 0
        for _ in range(SIMULATION_ROUNDS):
            if not len(numpy.unique(selected_cards[_])) == len(selected_cards[_]):
                ties += 1
                continue
            new_board_cards = board_cards + [deck[i] for i in selected_cards[_,:5-street]]
            revealed_cards = my_cards + new_board_cards
            opp_cards = [deck[i] for i in selected_cards[_,5-street:]]
            my_score = eval7.evaluate(revealed_cards)
            opp_score = eval7.evaluate(opp_cards + new_board_cards)
            wins += (my_score > opp_score)
            bounty_wins += ((my_score > opp_score) and (my_bounty in map(lambda c: ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')[c.rank], revealed_cards)))
            ties += (my_score == opp_score)
            
        return wins/(SIMULATION_ROUNDS-ties), bounty_wins/(SIMULATION_ROUNDS-ties)