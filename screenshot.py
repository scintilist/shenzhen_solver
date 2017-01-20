""" Run this script to save a screenshot of the board, the first arg is the file name to save as. """
from lib import gui
from sys import argv
from time import sleep

if __name__ == '__main__':
    ''' Get the board image. '''
    gui.find_window()
    sleep(1)
    board_image = gui.get_board_image()
    board_image.save(argv[1])
