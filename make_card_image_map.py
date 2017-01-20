""" Run this script to map the card images to card values.
    To use:
    1) Run SHENZHEN I/O solitare in a window with at least 1440x900 resolution
    2) Use the screenshot script to save screenshots of solitare boards
    3) Save board text files and fill them in to match the board images, saving each with the same file name
    4) Run this script, which will generate the card image map from
       all pairs of board and image files in the 'card_image_data' folder
"""
from PIL import Image

from lib.solver import Deck, Board
import glob
import os.path as path


if __name__ == '__main__':
    # Loop through all pairs of board text and image files
    txt = {path.splitext(f)[0] for f in glob.glob('card_image_data/*.txt')}
    png = {path.splitext(f)[0] for f in glob.glob('card_image_data/*.png')}
    for fn in txt.intersection(png):

        # Load the board from text file
        board = Board()
        with open(fn + '.txt', 'r') as f:
            board.from_string(f.read())

        # Load the board image
        board_image = Image.open(fn + '.png')

        # Update the card image map
        Deck.update_card_image_map(board_image, board)

    # Save the card image map
    Deck.save_card_image_map('card_image_data/card_images.p')

    # Load the card image map
    Deck.load_card_image_map('card_image_data/card_images.p')

    # Show the card image map
    Deck.show_card_image_map()
