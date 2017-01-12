""" Run this script to save a screenshot of the board """
from PIL import Image

from lib import solver
from lib import gui

# Save image to the path BOARD_IMAGE_FN
BOARD_IMAGE_FN = 'board_image.png'


if __name__ == '__main__':
    ''' Get the board image. ''':
    board_image = gui.get_live_board_image(gui.get_window())
    board_image.save(BOARD_IMAGE_FN)