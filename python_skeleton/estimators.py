import eval7
import numpy

class EquityEstimator:

    def estimate(self, my_cards, board_cards, my_bounty):
        pass
    
class MonteCarloEstimator(EquityEstimator):

    def estimate(self, my_cards, board_cards, my_bounty):
        # doubles are split into o and s, and two entries for each pair of ranks/suits to account for order
        hand_strength =  {
            'AAo': 85, 'AAs': 85, 'AKs': 67, 'AQs': 66, 'AJs': 65, 'ATs': 65, 'A9s': 63, 'A8s': 62, 'A7s': 61, 'A6s': 60,
            'A5s': 60, 'A4s': 59, 'A3s': 58, 'A2s': 57, 'AKo': 65, 'KKo': 82, 'KKs': 82, 'KQs': 63, 'KJs': 62, 'KTs': 60,
            'K9s': 58, 'K8s': 57, 'K7s': 57, 'K6s': 56, 'K5s': 55, 'K4s': 54, 'K3s': 53, 'K2s': 52, 'AQo': 64, 'KQo': 60,
            'QQo': 80, 'QQs': 80, 'QJs': 60, 'QTs': 59, 'Q9s': 58, 'Q8s': 56, 'Q7s': 54, 'Q6s': 54, 'Q5s': 53, 'Q4s': 52,
            'Q3s': 51, 'Q2s': 50, 'AJo': 64, 'KJo': 58, 'QJo': 55, 'JJo': 75, 'JJs': 75, 'JTs': 56, 'J9s': 54, 'J8s': 52,
            'J7s': 50, 'J6s': 50, 'J5s': 49, 'J4s': 48, 'J3s': 47, 'J2s': 46, 'ATo': 63, 'KTo': 57, 'QTo': 55, 'JTo': 53,
            'TTo': 72, 'TTs': 72, 'T9s': 54, 'T8s': 52, 'T7s': 51, 'T6s': 49, 'T5s': 47, 'T4s': 46, 'T3s': 45, 'T2s': 44,
            'A9o': 61, 'K9o': 58, 'Q9o': 55, 'J9o': 51, 'T9o': 51, '99o': 72, '99s': 72, '98s': 51, '97s': 49, '96s': 47,
            '95s': 46, '94s': 44, '93s': 43, '92s': 42, 'A8o': 60, 'K8o': 56, 'Q8o': 54, 'J8o': 51, 'T8o': 50, '98o': 50,
            '88o': 69, '88s': 69, '87s': 48, '86s': 46, '85s': 45, '84s': 44, '83s': 41, '82s': 41, 'A7o': 59, 'K7o': 55,
            'Q7o': 52, 'J7o': 49, 'T7o': 47, '97o': 47, '87o': 45, '77o': 66, '77s': 66, '76s': 43, '75s': 41, '74s': 41,
            '73s': 40, '72s': 38, 'A6o': 58, 'K6o': 54, 'Q6o': 51, 'J6o': 47, 'T6o': 45, '96o': 45, '86o': 43, '76o': 41,
            '66o': 63, '66s': 63, '65s': 43, '64s': 41, '63s': 38, '62s': 37, 'A5o': 57, 'K5o': 53, 'Q5o': 50, 'J5o': 46,
            'T5o': 44, '95o': 43, '85o': 41, '75o': 41, '65o': 40, '55o': 55, '55s': 55, '54s': 40, '53s': 39, '52s': 38,
            'A4o': 56, 'K4o': 52, 'Q4o': 49, 'J4o': 45, 'T4o': 43, '94o': 42, '84o': 40, '74o': 39, '64o': 38, '54o': 37,
            '44o': 40, '44s': 40, '43s': 38, '42s': 37, 'A3o': 55, 'K3o': 51, 'Q3o': 48, 'J3o': 44, 'T3o': 42, '93o': 41,
            '83o': 40, '73o': 38, '63o': 37, '53o': 36, '43o': 35, '33o': 33, '33s': 33, '32s': 35, 'A2o': 55, 'K2o': 50,
            'Q2o': 47, 'J2o': 43, 'T2o': 41, '92o': 40, '82o': 39, '72o': 38, '62o': 37, '52o': 36, '42o': 35, '32o': 34,
            '22o': 22, '22s': 22, 'KAs': 67, 'QAs': 66, 'JAs': 65, 'TAs': 65, '9As': 63, '8As': 62, '7As': 61, '6As': 60,
            '5As': 60, '4As': 59, '3As': 58, '2As': 57, 'KAo': 65, 'QKs': 63, 'JKs': 62, 'TKs': 60, '9Ks': 58, '8Ks': 57,
            '7Ks': 57, '6Ks': 56, '5Ks': 55, '4Ks': 54, '3Ks': 53, '2Ks': 52, 'QAo': 64, 'QKo': 60, 'JQs': 60, 'TQs': 59,
            '9Qs': 58, '8Qs': 56, '7Qs': 54, '6Qs': 54, '5Qs': 53, '4Qs': 52, '3Qs': 51, '2Qs': 50, 'JAo': 64, 'JKo': 58,
            'JQo': 55, 'TJs': 56, '9Js': 54, '8Js': 52, '7Js': 50, '6Js': 50, '5Js': 49, '4Js': 48, '3Js': 47, '2Js': 46,
            'TAo': 63, 'TKo': 57, 'TQo': 55, 'TJo': 53, '9Ts': 54, '8Ts': 52, '7Ts': 51, '6Ts': 49, '5Ts': 47, '4Ts': 46,
            '3Ts': 45, '2Ts': 44, '9Ao': 61, '9Ko': 58, '9Qo': 55, '9Jo': 51, '9To': 51, '89s': 51, '79s': 49, '69s': 47,
            '59s': 46, '49s': 44, '39s': 43, '29s': 42, '8Ao': 60, '8Ko': 56, '8Qo': 54, '8Jo': 51, '8To': 50, '89o': 50,
            '78s': 48, '68s': 46, '58s': 45, '48s': 44, '38s': 41, '28s': 41, '7Ao': 59, '7Ko': 55, '7Qo': 52, '7Jo': 49,
            '7To': 47, '79o': 47, '78o': 45, '67s': 43, '57s': 41, '47s': 41, '37s': 40, '27s': 38, '6Ao': 58, '6Ko': 54,
            '6Qo': 51, '6Jo': 47, '6To': 45, '69o': 45, '68o': 43, '67o': 41, '56s': 43, '46s': 41, '36s': 38, '26s': 37,
            '5Ao': 57, '5Ko': 53, '5Qo': 50, '5Jo': 46, '5To': 44, '59o': 43, '58o': 41, '57o': 41, '56o': 40, '45s': 40,
            '35s': 39, '25s': 38, '4Ao': 56, '4Ko': 52, '4Qo': 49, '4Jo': 45, '4To': 43, '49o': 42, '48o': 40, '47o': 39,
            '46o': 38, '45o': 37, '34s': 38, '24s': 37, '3Ao': 55, '3Ko': 51, '3Qo': 48, '3Jo': 44, '3To': 42, '39o': 41,
            '38o': 40, '37o': 38, '36o': 37, '35o': 36, '34o': 35, '23s': 35, '2Ao': 55, '2Ko': 50, '2Qo': 47, '2Jo': 43,
            '2To': 41, '29o': 40, '28o': 39, '27o': 38, '26o': 37, '25o': 36, '24o': 35, '23o': 34
        }


        SIMULATION_ROUNDS = 1000
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