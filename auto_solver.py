""" This script solves the Shenzhen solitare game in real time, using pyautogui to move the mouse and click """

import pyautogui
from time import perf_counter

from lib.solver import Solver, Board, Deck
from lib import gui

if __name__ == '__main__':
    ''' Load the card image map into the deck '''
    deck = Deck()
    deck.load_card_image_map('card_image_data/card_images.p')

    ''' Get the board image. '''
    w = gui.get_window()
    if not w:
        raise LookupError('Shenzhen I/O window not found')
    board_image = gui.get_live_board_image(w)


    ''' Load the board from the image '''
    board = Board()
    board.from_image(board_image)

    ''' Solve the board '''
    solver = Solver()
    solver.timeout = 0

    start = perf_counter()
    solved = solver.solve(board)
    duration = perf_counter() - start
    print('Board {} in {:.3f} seconds after {} boards tested'.format(solved, duration, solver.count))

    ''' Execute the solution using pyautogui '''
    # TODO
    # click the window to make it active
    x, y = gui.get_window_xy(w)
    pyautogui.moveTo(x+1, y+1, duration=0.5)
    pyautogui.click()

    for move in solver.list_moves():
        start, end = move
        x, y = gui.get_window_xy(w)
        pyautogui.moveTo(start[0]+x, start[1]+y, duration=0.5)
        pyautogui.dragTo(end[0]+x, end[1]+y, button='left', duration=0.5)
        print('({}, {}) to ({}, {})'.format(*move[0], *move[1]))
