from gi.repository import Gtk, Wnck
import pyautogui
import time


def get_window():
    """ Get the shenzhen solitare game window """
    # Find the SHENZHEN I/O Window and bring it to the front
    Gtk.main_iteration()
    screen = Wnck.Screen.get_default()
    screen.force_update()

    w = None
    for window in screen.get_windows():
        if window.get_name() == 'SHENZHEN I/O':
            if not window.is_active():
                window.activate(int(time.time()))
            w = window
            break
    return w


def get_window_xy(w):
    """ Get the x,y coordinates of the top left corner of the shenzhen solitare game window """
    xp, yp, width, height = w.get_client_window_geometry()
    return xp, yp


def get_live_board_image(w):
    """ Get an image of the shenzhen solitare game board """
    xp, yp, width, height = w.get_client_window_geometry()
    image = pyautogui.screenshot()
    return image.crop((xp, yp, xp + width, yp + height))


def correlation(im1, im2):
    """ Calculate the correlation between the 2 PIL images
        The correlation is 1 - normalized pixel rms error
        A white image and black image have a correlation of 0, and identical images have a correlation of 1
    """
    if im1.size != im2.size or im1.mode != im2.mode:
        raise ValueError('Images are different sizes.')

    square_error_sum = 0
    im1_data = im1.tobytes()
    im2_data = im2.tobytes()
    for i in range(len(im1_data)):
        square_error_sum += (im1_data[i] - im2_data[i])**2
    mean_square_error = square_error_sum / len(im1_data)
    rms_norm = mean_square_error ** 0.5 / 255
    return 1 - rms_norm