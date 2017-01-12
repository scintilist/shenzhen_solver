""" This script solves the Shenzhen solitare game in real time, using pyautogui to move the mouse and click """

import pyautogui
from time import perf_counter, sleep

from lib.solver import Solver, Board, Deck, Move
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
    solver = Solver()
    solver.timeout = 0

    start = perf_counter()
    solved = solver.solve(board)
    duration = perf_counter() - start
    print('Board {} in {:.3f} seconds after {} boards tested'.format(solved, duration, solver.count))
    print('Solution takes {} moves.'.format(len(solver.board_list)))

    ''' Execute the solution using pyautogui '''
    # Print all the moves
    for i, move in enumerate(solver.moves()):
        print(solver.board_list[i])
        print(move)
    print(solver.board_list[i+1])

    # click the window to make it active
    x, y = gui.get_window_xy(w)
    pyautogui.moveTo(x+1, y+1, duration=0.5)
    pyautogui.click()

    Move.window = w
    Move.verify = False
    for move in solver.moves():
        print('Attempting: ' + str(move))
        move.exec()
