""" Run this script to map the card images to card values.
    To use:
    1) Run SHENZHEN I/O solitare in a window at 1440x900 resolution
        (making sure all cards are within the screen boarders)
    2) fill in and save the "board.txt" file with the cards as they appear in the current game
    3) Run this script, which will match each card to it's image, and then save the resulting map in "card_images.p"

    The script can also be modified to use an image from a file, instead of a live image.
"""
from PIL import Image

from lib import solver
from lib import gui

# True to get a live image of the solitare board, False to use a saved image at the path BOARD_IMAGE_FN
LIVE = False
BOARD_IMAGE_FN = 'board_image.png'


if __name__ == '__main__':
    ''' Load board.txt '''
    board = solver.Board()
    with open('board.txt', 'r') as f:
        board.from_string(f.read())

    ''' Get the board image. '''
    if LIVE:
        board_image = gui.get_live_board_image()
    else:
        board_image = Image.open(BOARD_IMAGE_FN)

    ''' Map cards to card images. '''
    card_image_map = solver.Deck.create_card_image_map(board_image, board)

    ''' Save the card image map. '''
    with open('card_images.p', 'wb') as card_image_file:
        card_image_file.write(card_image_map)
