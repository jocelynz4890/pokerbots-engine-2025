import eval7
import numpy

class EquityEstimator:

    def estimate(self, my_cards, board_cards, my_bounty):
        pass
    
class MonteCarloEstimator(EquityEstimator):

    def estimate(self, my_cards, board_cards, my_bounty):
        # doubles are split into o and s, and two entries for each pair of ranks/suits to account for order
        hand_strength = {'AAo': 85, 'AAs': 85, 'AKs': 67, 'AQs': 66, 'AJs': 65, 'ATs': 65, 'A9s': 63, 'A8s': 62, 'A7s': 61, 'A6s': 60, 'A5s': 60, 'A4s': 59, 'A3s': 58, 'A2s': 57, 'AKo': 65, 'KKo': 82, 'KKs': 82, 'KQs': 63, 'KJs': 63, 'KTs': 62, 'K9s': 60, 'K8s': 58, 'K7s': 57, 'K6s': 57, 'K5s': 56, 'K4s': 55, 'K3s': 54, 'K2s': 53, 'AQo': 64, 'KQo': 61, 'QQo': 80, 'QQs': 80, 'QJs': 60, 'QTs': 59, 'Q9s': 58, 'Q8s': 56, 'Q7s': 54, 'Q6s': 54, 'Q5s': 53, 'Q4s': 52, 'Q3s': 51, 'Q2s': 50, 'AJo': 64, 'KJo': 61, 'QJo': 60, 'JJo': 77, 'JJs': 77, 'JTs': 57, 'J9s': 56, 'J8s': 54, 'J7s': 52, 'J6s': 50, 'J5s': 50, 'J4s': 49, 'J3s': 48, 'J2s': 47, 'ATo': 63, 'KTo': 60, 'QTo': 57, 'JTo': 55, 'TTo': 75, 'TTs': 75, 'T9s': 54, 'T8s': 52, 'T7s': 51, 'T6s': 49, 'T5s': 47, 'T4s': 46, 'T3s': 46, 'T2s': 45, 'A9o': 61, 'K9o': 58, 'Q9o': 55, 'J9o': 53, 'T9o': 51, '99o': 72, '99s': 72, '98s': 51, '97s': 49, '96s': 47, '95s': 46, '94s': 44, '93s': 43, '92s': 42, 'A8o': 60, 'K8o': 56, 'Q8o': 54, 'J8o': 51, 'T8o': 50, '98o': 48, '88o': 69, '88s': 69, '87s': 48, '86s': 46, '85s': 44, '84s': 43, '83s': 41, '82s': 40, 'A7o': 59, 'K7o': 55, 'Q7o': 52, 'J7o': 50, 'T7o': 48, '97o': 46, '87o': 45, '77o': 66, '77s': 66, '76s': 45, '75s': 44, '74s': 42, '73s': 40, '72s': 38, 'A6o': 58, 'K6o': 54, 'Q6o': 51, 'J6o': 48, 'T6o': 46, '96o': 44, '86o': 43, '76o': 42, '66o': 63, '66s': 63, '65s': 43, '64s': 41, '63s': 39, '62s': 38, 'A5o': 58, 'K5o': 53, 'Q5o': 50, 'J5o': 47, 'T5o': 44, '95o': 43, '85o': 41, '75o': 40, '65o': 40, '55o': 60, '55s': 60, '54s': 41, '53s': 40, '52s': 38, 'A4o': 57, 'K4o': 52, 'Q4o': 49, 'J4o': 46, 'T4o': 43, '94o': 41, '84o': 39, '74o': 38, '64o': 40, '54o': 38, '44o': 57, '44s': 57, '43s': 39, '42s': 37, 'A3o': 56, 'K3o': 51, 'Q3o': 48, 'J3o': 45, 'T3o': 42, '93o': 40, '83o': 37, '73o': 37, '63o': 36, '53o': 36, '43o': 35, '33o': 54, '33s': 54, '32s': 36, 'A2o': 55, 'K2o': 50, 'Q2o': 47, 'J2o': 44, 'T2o': 42, '92o': 39, '82o': 37, '72o': 35, '62o': 34, '52o': 34, '42o': 33, '32o': 32, '22o': 50, '22s': 50, 'KAs': 67, 'QAs': 66, 'JAs': 65, 'TAs': 65, '9As': 63, '8As': 62, '7As': 61, '6As': 60, '5As': 60, '4As': 59, '3As': 58, '2As': 57, 'KAo': 65, 'QKs': 63, 'JKs': 63, 'TKs': 62, '9Ks': 60, '8Ks': 58, '7Ks': 57, '6Ks': 57, '5Ks': 56, '4Ks': 55, '3Ks': 54, '2Ks': 53, 'QAo': 64, 'QKo': 61, 'JQs': 60, 'TQs': 59, '9Qs': 58, '8Qs': 56, '7Qs': 54, '6Qs': 54, '5Qs': 53, '4Qs': 52, '3Qs': 51, '2Qs': 50, 'JAo': 64, 'JKo': 61, 'JQo': 60, 'TJs': 57, '9Js': 56, '8Js': 54, '7Js': 52, '6Js': 50, '5Js': 50, '4Js': 49, '3Js': 48, '2Js': 47, 'TAo': 63, 'TKo': 60, 'TQo': 57, 'TJo': 55, '9Ts': 54, '8Ts': 52, '7Ts': 51, '6Ts': 49, '5Ts': 47, '4Ts': 46, '3Ts': 46, '2Ts': 45, '9Ao': 61, '9Ko': 58, '9Qo': 55, '9Jo': 53, '9To': 51, '89s': 51, '79s': 49, '69s': 47, '59s': 46, '49s': 44, '39s': 43, '29s': 42, '8Ao': 60, '8Ko': 56, '8Qo': 54, '8Jo': 51, '8To': 50, '89o': 48, '78s': 48, '68s': 46, '58s': 44, '48s': 43, '38s': 41, '28s': 40, '7Ao': 59, '7Ko': 55, '7Qo': 52, '7Jo': 50, '7To': 48, '79o': 46, '78o': 45, '67s': 45, '57s': 44, '47s': 42, '37s': 40, '27s': 38, '6Ao': 58, '6Ko': 54, '6Qo': 51, '6Jo': 48, '6To': 46, '69o': 44, '68o': 43, '67o': 42, '56s': 43, '46s': 41, '36s': 39, '26s': 38, '5Ao': 58, '5Ko': 53, '5Qo': 50, '5Jo': 47, '5To': 44, '59o': 43, '58o': 41, '57o': 40, '56o': 40, '45s': 41, '35s': 40, '25s': 38, '4Ao': 57, '4Ko': 52, '4Qo': 49, '4Jo': 46, '4To': 43, '49o': 41, '48o': 39, '47o': 38, '46o': 40, '45o': 38, '34s': 39, '24s': 37, '3Ao': 56, '3Ko': 51, '3Qo': 48, '3Jo': 45, '3To': 42, '39o': 40, '38o': 37, '37o': 37, '36o': 36, '35o': 36, '34o': 35, '23s': 36, '2Ao': 55, '2Ko': 50, '2Qo': 47, '2Jo': 44, '2To': 42, '29o': 39, '28o': 37, '27o': 35, '26o': 34, '25o': 34, '24o': 33, '23o': 32}


        SIMULATION_ROUNDS = 50
        ranks = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
        suits = ('c', 'd', 'h', 's')
        my_cards = list(map(eval7.Card, my_cards))
        board_cards = list(map(eval7.Card, board_cards))
        revealed_cards = my_cards + board_cards
        street = len(board_cards)
        
        if not street:
            hand_ranks = [ranks[c.rank] for c in my_cards]
            prob = hand_strength["".join(hand_ranks + ["s" if suits[my_cards[0].suit] == suits[my_cards[0].suit] else "o"])]/100
            return prob, prob if my_bounty in hand_ranks else 1-(48/52)**5
        
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
            bounty_wins += ((my_score > opp_score) and (my_bounty in map(lambda c: ranks[c.rank], revealed_cards)))
            ties += (my_score == opp_score)
            
        if ties == SIMULATION_ROUNDS:
            return 0, 0
        return wins/(SIMULATION_ROUNDS-ties), bounty_wins/(SIMULATION_ROUNDS-ties)