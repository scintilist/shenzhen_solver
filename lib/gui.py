from gi.repository import Gtk, Wnck
import pyautogui
import time


def get_live_board_image():
    """ Get an image of the shenzhen solitare game board """
    # Find the SHENZHEN I/O Window, bring it to the front, and take a screenshot
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
    if not w:
        return None # Shenzhen I/O window not found.

    xp, yp, width, height = w.get_client_window_geometry()
    #print('Window at ({}px,{}px) and has dimensions {}px X {}px'.format(xp, yp, width, height))

    # Delay to allow time for the window to make it to the front
    time.sleep(.5)

    image = pyautogui.screenshot()
    return image.crop((xp, yp, xp + width, yp + height))