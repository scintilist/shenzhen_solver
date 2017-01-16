""" Run this script to save a screenshot of the board """
from lib import gui

# Save image to the path BOARD_IMAGE_FN
BOARD_IMAGE_FN = 'board_image.png'


if __name__ == '__main__':
    ''' Get the board image. '''
    gui.find_window()
    board_image = gui.get_board_image()
    board_image.save(BOARD_IMAGE_FN)
