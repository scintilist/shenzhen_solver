""" This script solves the Shenzhen solitare game in real time, using pyautogui to move the mouse and click """

from PIL import Image
from time import perf_counter

from lib.solver import Solver, Board, Deck
from lib import gui

# True to get a live image of the solitare board, False to use a saved image at the path BOARD_IMAGE_FN
LIVE = True
BOARD_IMAGE_FN = 'board_image.png'


if __name__ == '__main__':
    ''' Load the card image map into the deck '''
    deck = Deck()
    deck.load_card_image_map('card_images.p')

    ''' Get the board image. '''
    if LIVE:
        board_image = gui.get_live_board_image()
    else:
        board_image = Image.open(BOARD_IMAGE_FN)

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
