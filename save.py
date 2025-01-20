import pickle
import os

from test_bot.player import pickle_file_path, main_file_path

# run this after training using table1/table2/table3 to sync with main table by aggregating results

with open(pickle_file_path, 'rb') as file:
    new_table = pickle.load(file) # new file that contains new qtable from training
print("new table loaded")
    
with open(main_file_path, 'rb') as file:
    table = pickle.load(file) # old qtable file to add result of new_table to
print("main table loaded")

OLD_WEIGHT = 0.5
NEW_WEIGHT = 0.5 # should maybe increase this and decrease the other if num rounds trained >> a couple million
assert OLD_WEIGHT + NEW_WEIGHT == 1, "weights should sum to 1"


new_states = 0
reward_zero = 0 # num states that have reward==0 (they were just discovered)
new_action = 0 
for state, action_rewards in new_table.items():
    if state not in table:
        new_states += 1
        table[state] = {}
        
    for action, reward in action_rewards.items():
        if action in table[state]:
            if table[state][action] == 0:
                reward_zero += 1
                table[state][action] = reward
            else:
                # add change in reward
                table[state][action] = OLD_WEIGHT*table[state][action] + NEW_WEIGHT*reward
        else:
            new_action  += 1
            table[state][action] = reward

print(f"{new_states} new states discovered, \nfirst reward set for {reward_zero} previously discovered states, \n{new_action} new actions taken")
    

# sync with main
with open(main_file_path, 'wb') as file:
    pickle.dump(table, file)
print(f"main table {main_file_path} has been updated by {pickle_file_path}")

# pull (updated) main table to the extra table
with open(pickle_file_path, 'wb') as file:
    pickle.dump(table, file)
print(f"newly updated main table at {main_file_path} also added to {pickle_file_path}")