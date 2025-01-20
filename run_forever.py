import subprocess
import time

def main():
    n = 0
    while True:
        n = n + 1
        # user_input = input()
        # if user_input.lower() == 'n':
        print(f'Starting round {n}')

        process = subprocess.run(["python3", "engine.py"], capture_output=True, text=True)
        
        # time.sleep(1)

if __name__ == "__main__":
    main()