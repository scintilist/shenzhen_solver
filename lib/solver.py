import unittest
import cProfile

from time import perf_counter, sleep
from functools import reduce

from random import shuffle, seed
from collections import namedtuple
import pickle
from PIL import Image, ImageDraw, ImageFont
import pyautogui

from lib import gui
import warnings

from copy import copy


""" Constants """
COLUMNS = 8              # Number of Columns in the main board
ROWS = 5                 # Number of rows in the main board
DRAGONS = 4              # Size of each set of dragons
SUITS = ['R', 'G', 'B']  # Suit iterable
VALUES = range(1, 10)    # Value iterable

# Card classes, uses named tuples to be fast and lightweight
Flower = namedtuple('Flower', [])
Dragon = namedtuple('Dragon', ['suit'])
Number = namedtuple('Number', ['suit', 'value'])


class Deck:
    """ Class for a deck of cards """
    card_images = {}

    @staticmethod
    def show_card_image_map():
        im_map = Image.new('RGB', (220, len(Deck.card_images)*25), (0xA0, 0xA0, 0xA0))
        draw = ImageDraw.Draw(im_map)
        font = ImageFont.truetype('Ubuntu-B.ttf', size=14)
        y = 0
        for card, im in sorted(Deck.card_images.items()):
            im_map.paste(im, (0, y, 20, y+20))
            draw.text((25, y), str(card), (0, 0, 0), font)
            y += 25
        im_map.show()

    @staticmethod
    def update_card_image_map(board_image, board):
        for card, x, y in board.card_coordinates():
            Deck.card_images[card] = board_image.crop((x, y, x + 20, y + 20))

    @staticmethod
    def save_card_image_map(fn):
        """ Save the card images to a pickle file """
        if len(Deck.card_images) < 32:
            warnings.warn('Incomplete map, only {} out of 32 cards found.'.format(len(Deck.card_images)))

        data = {card: {'mode': im.mode, 'size': im.size, 'data': im.tobytes()} for card, im in Deck.card_images.items()}
        with open(fn, 'wb') as f:
            f.write(pickle.dumps(data))

    @staticmethod
    def load_card_image_map(fn):
        """ Load the card images from the pickle file. """
        with open(fn, 'rb') as file:
            p_dict = pickle.load(file)
        Deck.card_images = {card: Image.frombytes(im['mode'], im['size'], im['data']) for card, im in p_dict.items()}

    @staticmethod
    def card_to_str(card):
        """ Prints the string representation of the card. """
        if isinstance(card, Flower):
            return 'F  '
        if isinstance(card, Dragon):
            return 'D' + card.suit + ' '
        if isinstance(card, Number):
            return 'N' + card.suit + str(card.value)
        raise TypeError

    @staticmethod
    def card_from_string(string):
        """ Return the card that is represented by the string.
            Raises KeyError if the string doesn't match a card
        """
        card, suit, value = string
        if card == 'F':
            return Flower()
        if card == 'D':
            return Dragon(suit)
        if card == 'N':
            return Number(suit, int(value))
        raise LookupError

    @staticmethod
    def card_from_image(image):
        if not Deck.card_images:
            raise LookupError('Deck does not have a card map.')
        for card, card_image in Deck.card_images.items():
            if gui.correlation(image, card_image) > 0.99:
                return copy(card)
        raise LookupError


# Space classes
class Space:
    def __init__(self, i, space=None):
        if space:
            self.cards = space.cards[:]
            self.key = space.key
            self.hash = space.hash
        else:
            self.cards = []
            self.update()
        self.i = i

    def append(self, card):
        self.cards.append(card)
        self.update()

    def extend(self, cards):
        self.cards.extend(cards)
        self.update()

    def pop(self, n=1):
        self.cards, cards = self.cards[:-n], self.cards[-n:]
        self.update()
        return cards

    def update(self):
        self.key = tuple(self.cards)
        self.hash = hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return self.hash

    def __str__(self):
        return '{}[{}]'.format(self.__class__.__name__, self.i)


class FlowerSpace(Space):
    def xy(self, count=0):
        return 738, 45


class Goal(Space):
    def xy(self, count=0):
        return self.i*152 + 930, 45


class Free(Space):
    def xy(self, count=0):
        return self.i*152 + 170, 45


class Main(Space):
    def xy(self, count=0):
        return self.i*152 + 170, (len(self.cards) - count) * 31 + 309


def vector_sum(*args):
    """ Add vectors of equal length, used to add xy coordinates together

    :param args: list of vectors to add together
    :return: vector sum of all args
    """
    return (*map(sum, zip(*args)),)


def distance(a, b):
    """ Return the distance between points a and b """
    return sum(map(lambda x, y: (x - y) ** 2, a, b)) ** 0.5


# Turn classes
class Turn:
    """ Base turn, can be one of several types, which have different initializers """

    window = None  # The game window
    speed = 5000  # Mouse move speed in pixels/second (default=5000)
    min_time = 0.1  # Minimum mouse move time in seconds (default=0.1)
    sleep_duration = 0.0  # Time to sleep after each mouse move (default=0.0)
    wait_duration = 0.25  # Seconds to wait while the automatic move executes (default=0.25)

    @staticmethod
    def absolute(point):
        """ Convert game window coordinates to absolute coordinates

        :param point: xy coordinates relative to the game window
        :return: absolute xy coordinates
        """
        x, y, width, height = Turn.window.get_client_window_geometry()
        return vector_sum((x, y), point)

    @staticmethod
    def generate(board):
        """ Generate all valid turns for the given board. Generation is stopped if there is a valid automatic move. """
        for turn in Auto.generate(board):
            yield turn
            return
        for turn in CollectDragons.generate(board):
            yield turn
        for turn in StackMove.generate(board):
            yield turn


class CollectDragons(Turn):
    """ Dragon collection turns. """
    button_xy = {'R': (648, 66), 'G': (648, 150), 'B': (648, 234)}

    def __init__(self, dst, srcs):
        """ Create the turn

        :param suit: suit of dragons to collect
        """
        self.dst = dst
        self.srcs = srcs

    def exec(self, verify=False):
        """ Execute the gui actions to bring the board to the next turn.

        :param verify: No effect
        """
        point = Turn.absolute(CollectDragons.button_xy[self.dst.cards[-1].suit])
        pyautogui.moveTo(*point, duration=Turn.min_time + distance(pyautogui.position(), point) / Turn.speed)
        sleep(Turn.sleep_duration)
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        sleep(Turn.wait_duration)

    def apply(self, board):
        """ Run the turn on the given board to generate the next board

        :param board: The board at the start of the turn
        :returns next_board: The new board after applying the turn
        """
        next_board = Board(board)
        next_board.space(self.dst).extend([next_board.space(src).pop()[0] for src in self.srcs])
        return next_board

    def __str__(self):
        return 'Collect dragons of suit {}'.format(self.dst.cards[-1].suit)

    @staticmethod
    def generate(board):
        """ Generate all valid dragon turns for the given board.

        :param board:
        :yield turn: Yields all dragon turns valid for the board
        """
        for dst in board.free:
            if len(dst.cards) == 1:
                if isinstance(dst.cards[-1], Dragon):
                    suit = dst.cards[-1].suit
                    srcs = []
                    for space in board.main + board.free:
                        if space.cards and isinstance(space.cards[-1], Dragon) and space.cards[-1].suit == suit:
                            srcs.append(space)
                    if len(srcs) == DRAGONS:
                        yield CollectDragons(dst, srcs)


class StackMove(Turn):
    """ Normal moves of stacks of cards from the main and free spaces to the main, free, and goal spaces """
    click_offset = (50, 10)  # Offset from the card image position to the card mouse click position

    def __init__(self, src, dst, count):
        """ Create the turn

        :param src: source space
        :param dst: destination space
        :param count: number of cards to move
        """
        self.src = src
        self.dst = dst
        self.count = count

    def exec(self, verify=False):
        """ Execute the gui actions to bring the board to the next turn.

        :param verify: True to verify that the board matches the expected state before executing the move
        """
        # Move to the start
        im_start = Turn.absolute(self.src.xy(self.count))
        start = vector_sum(im_start, StackMove.click_offset)
        pyautogui.moveTo(*start, duration=Turn.min_time + distance(pyautogui.position(), start) / Turn.speed)
        sleep(Turn.sleep_duration)

        if verify:
            # Verify the card
            image = pyautogui.screenshot()
            card_image = image.crop((*im_start, *vector_sum(im_start, (20, 20))))
            try:
                if self.src.cards[-self.count] != Deck.card_from_image(card_image):
                    raise LookupError
            except LookupError:
                raise RuntimeError('Drag move source card does not match the expected card image.')

        # Verify the mouse position
        if pyautogui.position() != start:
            raise RuntimeError('Mouse not in expected position')

        # Drag the card
        end = Turn.absolute(vector_sum(self.dst.xy(0), StackMove.click_offset))
        pyautogui.dragTo(*end, duration=Turn.min_time + distance(pyautogui.position(), end) / Turn.speed)
        sleep(Turn.sleep_duration)

    def apply(self, board):
        """ Run the turn on the given board to generate the next board

        :param board: The board at the start of the turn
        :returns next_board: The new board after applying the turn
        """
        next_board = Board(board)
        next_board.space(self.dst).extend(next_board.space(self.src).pop(self.count))
        return next_board

    def __str__(self):
        return 'Move {} cards from {} to {}'.format(self.count, self.src, self.dst)

    @staticmethod
    def generate(board):
        """ Generate all valid stack move turns for the given board.

        :param board:
        :yield turn: Yields all stack move turns valid for the board
        """
        # From main and free to goal
        for src in board.main + board.free:
            if src.cards and isinstance(src.cards[-1], Number):
                for dst in board.goal:
                    if dst.cards:
                        if src.cards[-1].suit == dst.cards[-1].suit and src.cards[-1].value == dst.cards[-1].value + 1:
                            yield StackMove(src, dst, 1)
                            break
                    elif src.cards[-1].value == VALUES[0]:
                        yield StackMove(src, dst, 1)
                        break

        # From main to free
        for src in board.main:
            if src.cards:
                for dst in board.free:
                    if not dst.cards:
                        yield StackMove(src, dst, 1)
                        break

        # From main to main
        for src in board.main:
            for n in range(1, len(src.cards) + 1):
                # Break when the stack becomes unmovable
                if n > 1 and not (
                        isinstance(src.cards[-n + 1], Number) and isinstance(src.cards[-n], Number) and
                        src.cards[-n + 1].suit != src.cards[-n].suit and
                        src.cards[-n + 1].value == src.cards[-n].value - 1):
                    break
                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst in board.main:
                    if src != dst:
                        if dst.cards:
                            if (isinstance(src.cards[-n], Number) and isinstance(dst.cards[-1], Number) and
                                    src.cards[-n].suit != dst.cards[-1].suit and
                                    src.cards[-n].value == dst.cards[-1].value - 1):
                                yield StackMove(src, dst, n)
                        elif not moved_to_empty:
                            moved_to_empty = True
                            yield StackMove(src, dst, n)

        # From free to main
        for src in board.free:
            if len(src.cards) == 1:
                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst in board.main:
                    if dst.cards:
                        if (isinstance(src.cards[-1], Number) and isinstance(dst.cards[-1], Number) and
                                src.cards[-1].suit != dst.cards[-1].suit and
                                src.cards[-1].value == dst.cards[-1].value - 1):
                            yield StackMove(src, dst, 1)
                    elif not moved_to_empty:
                        moved_to_empty = True
                        yield StackMove(src, dst, 1)


class Auto(StackMove):
    """ Automatic turns that do not require a mouse action.
        This includes moving the flower to the flower space, or balanced consecutive moves to the goal spaces.
    """

    def exec(self, verify=False):
        """ Execute the gui actions to bring the board to the next turn.

        :param verify: No effect
        """
        sleep(Turn.wait_duration)

    def __str__(self):
        return 'Move {} cards from {} to {} (Automatic)'.format(self.count, self.src, self.dst)

    @staticmethod
    def generate(board):
        """ Generate all valid automatic turns for the given board. Generation is stopped if there is a valid turn.

        :param board:
        :yield turn: Yields all automatic turns valid for the board
        """
        max_goal_value = reduce(lambda m, dst: min(m, dst.cards[-1].value) if dst.cards else 1, board.goal, 9) + 1
        for src in board.main + board.free:
            if src.cards:
                # To goal if the card is less than 2 greater than the lowest goal value
                if isinstance(src.cards[-1], Number):
                    for dst in board.goal:
                        if dst.cards:
                            if (src.cards[-1].suit == dst.cards[-1].suit and
                                    src.cards[-1].value == dst.cards[-1].value + 1 and
                                    src.cards[-1].value <= max_goal_value):
                                yield Auto(src, dst, 1)
                                return
                        else:
                            if src.cards[-1].value == VALUES[0]:
                                yield Auto(src, dst, 1)
                                return

                # Move the flower to the flower space
                if isinstance(src.cards[-1], Flower):
                    yield Auto(src, board.flower, 1)
                    return


# Board class
class Board:
    def __init__(self, board=None):
        """ Create the empty board, or copy the board. """
        if board:
            # Cards are only moved, never modified, so it is ok to copy card references, and not duplicate cards
            self.main = [Main(i, space) for i, space in enumerate(board.main)]
            self.free = [Free(i, space) for i, space in enumerate(board.free)]
            self.goal = [Goal(i, space) for i, space in enumerate(board.goal)]
            self.flower = FlowerSpace(0, board.flower)
        else:
            self.main = [Main(i) for i in range(COLUMNS)]
            self.free = [Free(i) for i in range(len(SUITS))]
            self.goal = [Goal(i) for i in range(len(SUITS))]
            self.flower = FlowerSpace(0)
        self.turn_generator = Turn.generate(self)

    def randomize(self):
        """ Create a random board. """
        # Make a deck of cards
        deck = []
        for suit in SUITS:
            for value in VALUES:
                deck.append(Number(suit, value))
            for i in range(DRAGONS):
                deck.append(Dragon(suit))
        deck.append(Flower())

        # Shuffle and lay out the 8 columns of 5 cards each
        shuffle(deck)
        self.main = []
        for i, cards in enumerate(zip(*[iter(deck)] * ROWS)):
            space = Main(i)
            space.extend(cards)
            self.main.append(space)

        # Create the free spaces, goal spaces, and flower space
        self.free = [Free(i) for i in range(len(SUITS))]
        self.goal = [Goal(i) for i in range(len(SUITS))]
        self.flower = FlowerSpace(0)

    def space(self, space):
        """ Get the space in the board, where the parameter space may have been from a different board """
        if type(space) == Main:
            return self.main[space.i]
        if type(space) == Goal:
            return self.goal[space.i]
        if type(space) == Free:
            return self.free[space.i]
        if type(space) == FlowerSpace:
            return self.flower

    def is_solved(self):
        """ Returns True if the board is solved. """
        # Detects solved boards by checking if the main columns are empty.
        # Assuming all rules were followed, this is only true for a solved board.
        return not any(main.cards for main in self.main)

    def __str__(self):
        """ String representation of the board. """
        s = ''
        # Free spaces, if there is a stack of cards, show the number stacked
        for free in self.free:
            try:
                s += '[' + Deck.card_to_str(free.cards[-1])
                s += ']' if len(free.cards) == 1 else str(len(free.cards))
            except IndexError:
                s += '[   ]'

        # Flower space
        try:
            s += '  [' + Deck.card_to_str(self.flower.cards[-1]) + ']   '
        except IndexError:
            s += '  [   ]   '

        # Goal spaces
        for goal in self.goal:
            try:
                s += '[' + Deck.card_to_str(goal.cards[-1]) + ']'
            except IndexError:
                s += '[   ]'
        s += '\n'

        # Main spaces
        empty = False
        row = 0
        while not empty:
            empty = True
            for main in self.main:
                try:
                    s += '[' + Deck.card_to_str(main.cards[row]) + ']'
                except IndexError:
                    s += '     '
                else:
                    empty = False
            row += 1
            s += '\n'
        return s

    def from_string(self, string):
        """ Loads the board from a string representation. """
        # Break the string by lines so it can be indexed by x/y coordinates.
        s = string.split('\n')

        # Load the free space cards.
        # If there is a stack of cards, the following character is the stack height
        self.free = []
        for i in range(len(SUITS)):
            space = Free(i)
            try:
                index = i * 5 + 1
                card = Deck.card_from_string(s[0][index:index+3])
                try:
                    count = int(s[0][index+3])
                except ValueError:
                    count = 1
                space.extend([card] * count)
            except (LookupError, ValueError):
                pass
            self.free.append(space)

        # Load the flower space card
        self.flower = FlowerSpace(0)
        try:
            index = i * 5 + 8
            self.flower.append(Deck.card_from_string(s[0][index:index+3]))
        except (LookupError, ValueError):
            pass

        # Load the goal cards. Generate the full stack down to the 1 card if it is a number
        self.goal = []
        for i in range(len(SUITS)):
            space = Goal(i)
            try:
                index = len(SUITS) * 5 + i * 5 + 11
                card = Deck.card_from_string(s[0][index:index+3])
                if isinstance(card, Number):
                    for value in range(1, card.value+1):
                        space.append(Number(card.suit, value))
                else:
                    space.append(card)
            except (LookupError, ValueError):
                pass
            self.goal.append(space)

        # Load the Main space cards
        self.main = []
        for i in range(COLUMNS):
            space = Main(i)
            try:
                row = 1
                while True:
                    index = i * 5 + 1
                    card = Deck.card_from_string(s[row][index:index+3])
                    space.append(card)
                    row += 1
            except (LookupError, ValueError, IndexError):
                pass
            self.main.append(space)

    def from_image(self, board_image):
        """ Loads the board from an image of the board """
        # Load the free space cards.
        self.free = []
        for col in range(len(SUITS)):
            space = Free(col)
            x, y = space.xy()
            try:
                space.append(Deck.card_from_image(board_image.crop((x, y, x + 20, y + 20))))
            except LookupError:
                pass
            self.free.append(space)

        # Load the flower space card
        self.flower = FlowerSpace(0)
        try:
            x, y = self.flower.xy()
            self.flower.append(Deck.card_from_image(board_image.crop((x, y, x + 20, y + 20))))
        except LookupError:
            pass

        # Load the goal cards.
        self.goal = []
        for col in range(len(SUITS)):
            space = Goal(col)
            x, y = space.xy()
            for shift in range(0, -9, -1):
                try:
                    card = Deck.card_from_image(board_image.crop((x, y+shift, x + 20, y+shift + 20)))
                    for value in range(1, card.value + 1):
                        space.append(Number(card.suit, value))
                    break
                except LookupError:
                    pass
            self.goal.append(space)

        # Load the Main space cards
        self.main = []
        for col in range(COLUMNS):
            space = Main(col)
            try:
                while True:
                    x, y = space.xy()
                    space.append(Deck.card_from_image(board_image.crop((x, y, x + 20, y + 20))))
            except (LookupError, IndexError):
                pass
            self.main.append(space)

        # Replace dragon stack placeholder cards in the free spaces with the stacks of dragons
        for suit in SUITS:
            # Count visible dragons of the suit (only need to check the main area and free spaces
            count = 0
            for space in self.main + self.free:
                for card in space.cards:
                    if isinstance(card, Dragon) and card.suit == suit:
                        count += 1

            # If there are missing dragons, replace the first suitless free space dragon with them
            if count < DRAGONS:
                for col, space in enumerate(self.free):
                    if space.cards and isinstance(space.cards[-1], Dragon) and space.cards[-1].suit not in SUITS:
                        self.free[col] = Free(col)
                        for i in range(DRAGONS):
                            self.free[col].append(Dragon(suit))
                        break

    def card_coordinates(self):
        """ Generator that yields all visible cards in the board along with their screen coordinate offsets. """
        # Free spaces
        for col, free in enumerate(self.free):
            i = 0
            while True:
                try:
                    card = free.cards[i]
                except IndexError:
                    break
                else:
                    yield (card, *free.xy())
                i += 1

        # Flower space
        try:
            card = self.flower.cards[-1]
        except IndexError:
            pass
        else:
            yield (card, *self.flower.xy())

        # Goal spaces
        for col, goal in enumerate(self.goal):
            try:
                card = goal.cards[-1]
            except IndexError:
                pass
            else:
                yield (card, *goal.xy())

        # Main spaces
        for col, main in enumerate(self.main):
            row = 1
            while True:
                try:
                    card = main.cards[-row]
                except IndexError:
                    break
                else:
                    yield (card, *main.xy(row))
                row += 1

    def key(self):
        return (frozenset(self.main),) + (frozenset(self.free),) + tuple(frozenset(self.goal),) + (self.flower,)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class Solve:
    """ Backtracking solver """

    def __init__(self, board, timeout=0):
        """ Solve the puzzle using backtracking.

        :param board: The board to be solved
        :param timeout: the solver timeout is seconds, if 0, will never time out
        """
        self.board = board          # Board to be solved
        self.timeout = 0

        self.turns = []             # List of turns in the solution
        self.boards = [board]       # List of boards in the solution
        self.board_cache = {board}  # Cache of all boards seen while solving

        self.duration = 0           # Time taken to solve the board in seconds
        self.result = ''            # Result of the last solution attempt

        if board.is_solved():
            self.result = 'solved'
            return

        start_time = perf_counter()
        while True:
            try:
                # Generate the next turn and board
                turn = next(self.boards[-1].turn_generator)
                board = turn.apply(self.boards[-1])
                if board not in self.board_cache:
                    self.board_cache.add(board)
                    self.boards.append(board)
                    self.turns.append(turn)
                    if board.is_solved():
                        self.result = 'solved'
                        break
                    if not len(self.board_cache) % 1000 and timeout:
                        if perf_counter() - start_time > timeout:
                            self.result = 'timed out'
                            break

            except StopIteration:
                # If there are no more boards to generate, then pop the board and continue from the previous
                self.boards.pop()
                if not self.boards:
                    # If there are no more boards to pop, then the board is unsolvable.
                    self.result = 'unsolvable'
                    break
                self.turns.pop()

        self.duration = perf_counter() - start_time

    def prune(self):
        """ Shorten the solution by merging and removing non-productive turns. """
        pass
        # TODO: Merge consecutive moves of the same card

        # TODO: Remove pairs of moves that do not affect the board state
        # If a card is moved, and then returns to it's state without ever having cards stacked on it
        # or the original source, then the set of moves can be removed.

        # At each move, scan ahead to see if the card ever ends up back in the same position
        # once found, check between that the original stack was never modified
        # check that the moved stack has not been modified in this time

    def exec(self, show=False, verify=False):
        """ Execute the solution in the game.

        :param show: If True, print each turn before execution
        :param verify: If True, verify the board before execution of each move
        """
        for turn in self.turns:
            if show:
                print('Attempting: ' + str(turn))
            turn.exec(verify=verify)

    def print(self):
        """ Print all the turns and boards in the solution """
        if self.result == 'solved':
            for i, turn in enumerate(self.turns):
                print(self.boards[i])
                print(turn)
            if self.boards:
                print(self.boards[-1])
            print('Solution takes {} turns.'.format(len(self.turns)))

        print('Board {} in {:.3f} seconds after {} boards tested'.format(
              self.result, self.duration, len(self.board_cache)))


class Tests(unittest.TestCase):
    def test_card_compare(self):
        self.assertEqual(Dragon(SUITS[0]), Dragon(SUITS[0]))
        self.assertNotEqual(Dragon(SUITS[0]), Dragon(SUITS[1]))
        self.assertNotEqual(Dragon(SUITS[0]), Flower())
        self.assertNotEqual(Dragon(SUITS[0]), Number(SUITS[0], 3))

    def test_board_compare(self):
        b = Board()
        b.randomize()
        b2 = Board(b)
        self.assertEqual(b, b2)
        b.main[1].extend(b.main[2].pop())
        self.assertNotEqual(b, b2)

        # Test free space symmetry
        b = Board()
        b.free[0].cards.append(Dragon(SUITS[0]))
        b.free[1].cards.append(Dragon(SUITS[1]))
        b2 = Board()
        b2.free[2].cards.append(Dragon(SUITS[0]))
        b2.free[0].cards.append(Dragon(SUITS[1]))
        self.assertEqual(b, b2)

        # Test main space symmetry
        b = Board()
        b.main[0].cards.append(Dragon(SUITS[0]))
        b.main[1].cards.append(Dragon(SUITS[1]))
        b2 = Board()
        b2.main[2].cards.append(Dragon(SUITS[0]))
        b2.main[0].cards.append(Dragon(SUITS[1]))
        self.assertEqual(b, b2)

    def test_board_hashing(self):
        b = Board()
        b.randomize()
        b2 = Board(b)
        self.assertTrue(len({b, b2}) == 1)
        b.main[1].extend(b.main[2].pop())
        self.assertTrue(len({b, b2}) == 2)

    def test_solved(self):
        b = Board()
        self.assertTrue(b.is_solved())
        b.randomize()
        self.assertFalse(b.is_solved())
        b = Board()
        b.main[4].cards.append(Flower())
        self.assertFalse(b.is_solved())

    def test_string_methods(self):
        """ Test that the board can be parsed to a string and back. """
        b = Board()
        b.randomize()
        b.free[0].extend(b.main[2].pop())
        b.free[1].extend([Dragon(SUITS[0])]*4)
        for i in range(1, 7):
            b.goal[1].append(Number(SUITS[1], i))
        b.flower.extend(b.main[4].pop())
        b2 = Board()
        b2.from_string(str(b))
        self.assertEqual(b, b2)


if __name__ == '__main__':
    if False:
        seed(47)
        board = Board()
        board.randomize()
        cProfile.run('Solve(board)')
        exit()

    seeds = [1, 2, 3, 4, 5, 6, 7, 9]
    move_count = 0
    for i in seeds:
        seed(i)
        board = Board()
        board.randomize()
        #print(board)
        solution = Solve(board)
        solution.prune()

        print('Seed {} {} in {:.3f} seconds after {} boards tested, takes {} moves'.format(
                i, solution.result, solution.duration, len(solution.board_cache), len(solution.boards)))
        move_count += len(solution.boards)
    print('Average solution takes {} moves.'.format(move_count//len(seeds)))
