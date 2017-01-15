""" This script solves the Shenzhen solitare game in real time, using pyautogui to move the mouse and click """

import pyautogui
from time import perf_counter, sleep

from lib.solver import Solve, Board, Deck, Turn
from lib import gui

if __name__ == '__main__':
    ''' Load the card image map into the deck '''
    deck = Deck()
    deck.load_card_image_map('card_image_data/card_images.p')

    ''' Get the board image. '''
    w = gui.get_window()
    if not w:
        raise LookupError('Shenzhen I/O window not found')
    sleep(0.5)
    board_image = gui.get_live_board_image(w)


    ''' Load the board from the image '''
    board = Board()
    board.from_image(board_image)

    ''' Solve the board '''
    solution = Solve(board, timeout=2)

    solution.prune()
    solution.print()

    ''' Execute the solution  if it was solved '''
    if solution.result == 'solved':
        Turn.window = w

        solution.exec(show=True, verify=False)

