import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Wnck', '3.0')
from gi.repository import Gtk, Wnck

import pyautogui
from time import sleep, time

# Minimum game window resolution
min_width = 1440
min_height = 900

# Reference to the SHENZHEN I/O game window
window = None

# Mouse movement settings
speed = 50000  # Mouse move speed in pixels/second (default=5000)
min_time = 0.1  # Minimum mouse move time in seconds (default=0.1)
sleep_duration = 0.0  # Time to sleep after each mouse move (default=0.0)


def vector_sum(*args):
    """ Add vectors of equal length, used to add xy coordinates together.

    :param args: list of vectors to add together
    :return: vector sum of all args
    """
    return (*map(sum, zip(*args)),)


def distance(a, b):
    """ Return the distance between points a and b. """
    return sum(map(lambda x, y: (x - y) ** 2, a, b)) ** 0.5


def absolute(point):
    """ Convert game window coordinates to absolute coordinates

    :param point: xy coordinates relative to the game window
    :return: absolute xy coordinates
    """
    return vector_sum(get_board_xy(), point)


def move_to(x, y):
    """ Move the mouse to the game window relative coordinates, following the Turn settings. """
    point = absolute((x, y))
    pyautogui.moveTo(*point, duration=min_time + distance(pyautogui.position(), point) / speed)
    sleep(sleep_duration)


def drag_to(x, y):
    """ Move the mouse to the game window relative coordinates, following the Turn settings. """
    point = absolute((x, y))
    pyautogui.dragTo(*point, duration=min_time + distance(pyautogui.position(), point) / speed)
    sleep(sleep_duration)


def click():
    """ Click the mouse at the current position. """
    pyautogui.mouseDown()
    pyautogui.mouseUp()


def find_window():
    """ Find the SHENZHEN I/O Window and bring it to the front. """
    global window
    window = None

    Gtk.main_iteration()
    screen = Wnck.Screen.get_default()
    screen.force_update()

    for w in screen.get_windows():
        if w.get_name() == 'SHENZHEN I/O':
            if not w.is_active():
                w.activate(int(time()))
            window = w
            return
    raise RuntimeError('Shenzhen I/O window not found')


def get_board_xy():
    """ Get the x,y coordinates of the top left corner of the shenzhen solitare game window. """
    xp, yp, width, height = window.get_client_window_geometry()
    xp += (width-min_width) // 2
    yp += (height-min_height) // 2
    return xp, yp


def get_board_image():
    """ Get an image of the shenzhen solitare game board. """
    image = pyautogui.screenshot()
    xp, yp = get_board_xy()
    return image.crop((xp, yp, xp + min_width, yp + min_height))


def get_card_image(image, x, y):
    """ Crop the board image to the 20 x 20 pixel card image at the given xy coordinates. """
    return image.crop((x, y, x + 20, y + 20))


def correlation(im1, im2):
    """ Calculate the correlation between the 2 PIL images.
        The correlation is 1 - normalized pixel rms error.
        A white image and black image have a correlation of 0, and identical images have a correlation of 1.
        Only compares 1 in 4 pixels for faster performance.
    """
    if im1.size != im2.size or im1.mode != im2.mode:
        raise ValueError('Images are different sizes.')

    square_error_sum = 0
    im1_data = im1.tobytes()
    im2_data = im2.tobytes()
    for i in range(0, len(im1_data), 4):
        square_error_sum += (im1_data[i] - im2_data[i])**2
    mean_square_error = square_error_sum / (len(im1_data)/4)
    rms_norm = mean_square_error ** 0.5 / 255
    return 1 - rms_norm
