import time
from pynput import keyboard


from src.main import get_number_of_cards

while True:
    print(get_number_of_cards())
    time.sleep(0.2)