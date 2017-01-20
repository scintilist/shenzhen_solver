import unittest
import cProfile

from collections import namedtuple
from copy import copy
from functools import reduce
from heapq import heappush, heappop
import pickle
from random import shuffle, seed
from time import perf_counter, sleep
import warnings

from PIL import Image, ImageDraw, ImageFont
import pyautogui

from lib import gui

""" Constants """
COLUMNS = 8              # Number of stack columns
ROWS = 5                 # Number of rows in the stacks
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
            self._update()
        self.i = i

    def append(self, card):
        self.cards.append(card)
        self._update()

    def extend(self, cards):
        self.cards.extend(cards)
        self._update()

    def pop(self, n=1):
        self.cards, cards = self.cards[:-n], self.cards[-n:]
        self._update()
        return cards

    def _update(self):
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


class Stacks(Space):
    def xy(self, count=0):
        return self.i*152 + 170, (len(self.cards) - count) * 31 + 309


# Turn classes
class Turn:
    """ Base turn, can be one of several types, which have different initializers """
    wait_duration = 0.25  # Seconds to wait while the automatic move executes (default=0.25)

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
        gui.move_to(*CollectDragons.button_xy[self.dst.cards[-1].suit])
        gui.click()
        sleep(Turn.wait_duration)

    def apply(self, board):
        """ Run the turn on the given board to generate the next board

        :param board: The board at the start of the turn
        :returns next_board: The new board after applying the turn
        """
        next_board = Board(board)
        next_board.turn = self
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
                    for space in board.stacks + board.free:
                        if space.cards and isinstance(space.cards[-1], Dragon) and space.cards[-1].suit == suit:
                            srcs.append(space)
                    if len(srcs) == DRAGONS:
                        yield CollectDragons(dst, srcs)


class StackMove(Turn):
    """ Normal moves of stacks of cards from the stack and free spaces to the stack, free, and goal spaces """
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
        start = gui.vector_sum(self.src.xy(self.count), StackMove.click_offset)
        gui.move_to(*start)

        if verify:
            # Verify the card
            card_image = gui.get_card_image(gui.get_board_image(), *self.src.xy(self.count))
            try:
                if self.src.cards[-self.count] != Deck.card_from_image(card_image):
                    raise LookupError
            except LookupError:
                raise RuntimeError('Drag move source card does not match the expected card image.')

        # Verify the mouse position
        if pyautogui.position() != gui.absolute(start):
            raise RuntimeError('Mouse not in expected position')

        # Drag the card
        end = gui.vector_sum(self.dst.xy(0), StackMove.click_offset)
        gui.drag_to(*end)

    def apply(self, board):
        """ Run the turn on the given board to generate the next board

        :param board: The board at the start of the turn
        :returns next_board: The new board after applying the turn
        """
        next_board = Board(board)
        next_board.turn = self
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
        # From stacks and free to goal
        for src in board.stacks + board.free:
            if src.cards and isinstance(src.cards[-1], Number):
                for dst in board.goal:
                    if dst.cards:
                        if src.cards[-1].suit == dst.cards[-1].suit and src.cards[-1].value == dst.cards[-1].value + 1:
                            yield StackMove(src, dst, 1)
                            break
                    elif src.cards[-1].value == VALUES[0]:
                        yield StackMove(src, dst, 1)
                        break

        # From stacks to free
        for src in board.stacks:
            if src.cards:
                for dst in board.free:
                    if not dst.cards:
                        yield StackMove(src, dst, 1)
                        break

        # From stacks to stacks
        # Move in order from tallest to shortest stack for each space
        for src in board.stacks:
            src.max_n = len(src.cards)
            for n in range(2, len(src.cards)+1):
                # Break when the stack becomes unmovable
                if not (isinstance(src.cards[-n + 1], Number) and isinstance(src.cards[-n], Number) and
                        src.cards[-n + 1].suit != src.cards[-n].suit and
                        src.cards[-n + 1].value == src.cards[-n].value - 1):
                    src.max_n = n - 1
                    break
            for n in range(src.max_n, 0, -1):
                # Move stacks from largest to smallest
                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst in board.stacks:
                    if src != dst:
                        if dst.cards:
                            if (isinstance(src.cards[-n], Number) and isinstance(dst.cards[-1], Number) and
                                    src.cards[-n].suit != dst.cards[-1].suit and
                                    src.cards[-n].value == dst.cards[-1].value - 1):
                                yield StackMove(src, dst, n)
                        elif not moved_to_empty:
                            moved_to_empty = True
                            yield StackMove(src, dst, n)

        # From free to stacks
        for src in board.free:
            if len(src.cards) == 1:
                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst in board.stacks:
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
        for src in board.stacks + board.free:
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
            self.stacks = [Stacks(i, space) for i, space in enumerate(board.stacks)]
            self.free = [Free(i, space) for i, space in enumerate(board.free)]
            self.goal = [Goal(i, space) for i, space in enumerate(board.goal)]
            self.flower = FlowerSpace(0, board.flower)
            self.previous_board = board
            self.turn_count = board.turn_count + 1
            self.cards_remaining = board.cards_remaining
        else:
            self.stacks = [Stacks(i) for i in range(COLUMNS)]
            self.free = [Free(i) for i in range(len(SUITS))]
            self.goal = [Goal(i) for i in range(len(SUITS))]
            self.flower = FlowerSpace(0)
            self.turn_count = 0
            self.cards_remaining = 0
        self.turn_generator = Turn.generate(self)
        self.score = self.turn_count + self.cards_remaining

    def calc_score(self):
        """ Calculate the board score, lower is better.
            Also updates the cards remaining count.
        """
        self.cards_remaining = sum(len(stack.cards) for stack in self.stacks)
        #self.score = self.turn_count + self.cards_remaining
        self.score = self.cards_remaining

    def is_solved(self):
        """ Returns True if the board is solved. """
        # Detects solved boards by checking if the stack columns are empty.
        # Assuming all rules were followed, this is only true for a solved board.
        # Must have called the score method before to update the cards remaining
        return not self.cards_remaining

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
        self.stacks = []
        for i, cards in enumerate(zip(*[iter(deck)] * ROWS)):
            space = Stacks(i)
            space.extend(cards)
            self.stacks.append(space)

        # Create the free spaces, goal spaces, and flower space
        self.free = [Free(i) for i in range(len(SUITS))]
        self.goal = [Goal(i) for i in range(len(SUITS))]
        self.flower = FlowerSpace(0)

    def next(self):
        """ Get the next board generated with a move from the current board. """
        return next(self.turn_generator).apply(self)

    def space(self, space):
        """ Get the space in the board, where the parameter space may have been from a different board """
        if type(space) == Stacks:
            return self.stacks[space.i]
        if type(space) == Goal:
            return self.goal[space.i]
        if type(space) == Free:
            return self.free[space.i]
        if type(space) == FlowerSpace:
            return self.flower

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
            for stack in self.stacks:
                try:
                    s += '[' + Deck.card_to_str(stack.cards[row]) + ']'
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
        self.stacks = []
        for i in range(COLUMNS):
            space = Stacks(i)
            try:
                row = 1
                while True:
                    index = i * 5 + 1
                    card = Deck.card_from_string(s[row][index:index+3])
                    space.append(card)
                    row += 1
            except (LookupError, ValueError, IndexError):
                pass
            self.stacks.append(space)

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
        self.stacks = []
        for col in range(COLUMNS):
            space = Stacks(col)
            try:
                while True:
                    x, y = space.xy()
                    space.append(Deck.card_from_image(board_image.crop((x, y, x + 20, y + 20))))
            except (LookupError, IndexError):
                pass
            self.stacks.append(space)

        # Replace dragon stack placeholder cards in the free spaces with the stacks of dragons
        for suit in SUITS:
            # Count visible dragons of the suit (only need to check the stack area and free spaces
            count = 0
            for space in self.stacks + self.free:
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
        # Free, goal, and flower spaces
        for space in self.free + self.goal + [self.flower]:
            for card in space.cards:
                yield (card, *space.xy())

        # Main spaces
        for space in self.stacks:
            for row, card in enumerate(space.cards):
                yield (card, *space.xy(len(space.cards) - row))

    def key(self):
        return (frozenset(self.stacks),) + (frozenset(self.free),) + tuple(frozenset(self.goal),) + (self.flower,)

    def __lt__(self, other):
        """ Used to order the boards in the heap by their scores, rather than their key values. """
        return self.score < other.score

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

        self.boards = [board]       # List of all active boards, sorted by their scores
        self.board_cache = {board}  # Cache of all boards seen while solving

        self.duration = 0           # Time taken to solve the board in seconds
        self.result = ''            # Result of the last solution attempt

        board.calc_score()
        if board.is_solved():
            self.result = 'solved'
            return

        start_time = perf_counter()
        while True:
            try:
                # Generate the next board from the current lowest scoring board
                board = self.boards[0].next()
                #print('board generated, count = {}'.format(len(self.board_cache)))
                if board not in self.board_cache:
                    self.board_cache.add(board)
                    board.calc_score()
                    heappush(self.boards, board)
                    #print('board pushed, len = {}'.format(len(self.boards)))

                    scores = [(board.score, board) for board in self.boards]
                    #print('board scores = {}'.format(str(scores)))
                    #print('board score = {}'.format(board.score))
                    #print('prev score = {}'.format(board.previous_board.score))
                    #print('score = {}, remaining = {}, turns = {}'.format(board.score, board.cards_remaining, board.turn_count))
                    #print(board)
                    #input()

                    if board.is_solved():
                        self.result = 'solved'
                        break
                    if timeout and perf_counter() - start_time > timeout:
                        self.result = 'timed out'
                        break

            except StopIteration:
                # If there are no more boards to generate, then pop the board and continue from the previous
                heappop(self.boards)
                #print('board popped, len = {}'.format(len(self.boards)))
                if not self.boards:
                    # If there are no more boards to pop, then the board is unsolvable.
                    self.result = 'unsolvable'
                    break

        # Create the turn and board lists
        self.boards = []
        self.turns = []
        try:
            while True:
                self.boards.append(board)
                self.turns.append(board.turn)
                board = board.previous_board
        except AttributeError:
            self.boards.reverse()
            self.turns.reverse()

        self.duration = perf_counter() - start_time

    def prune(self):
        """ Shorten the solution by merging and removing non-productive turns. """
        initial_length = len(self.turns)

        # Merge consecutive moves of the same card
        i = 0
        while i < len(self.turns) - 1:
            if (isinstance(self.turns[i], StackMove) and
                    isinstance(self.turns[i+1], StackMove) and
                    type(self.turns[i].dst) == type(self.turns[i+1].src) and
                    self.turns[i].dst.i == self.turns[i+1].src.i and
                    self.turns[i].count == self.turns[i+1].count):
                # print('Merged turns {} ({}) and {} ({})'.format(i, self.turns[i], i+1, self.turns[i+1]))
                self.turns[i].dst = self.turns[i+1].dst
                del self.turns[i+1]
            else:
                i += 1

        # Remove pairs of moves that do not affect the board state
        # If a card is moved, and then returns to it's state without ever having cards stacked on it
        # or the original source, then the set of moves can be removed.
        i = 0
        while i < len(self.turns):
            if isinstance(self.turns[i], StackMove):
                src = self.turns[i].src
                dst = self.turns[i].dst
                count = self.turns[i].count
                j = i + 1
                while j < len(self.turns):
                    if isinstance(self.turns[j], StackMove):
                        if (type(self.turns[j].src) == type(dst) and self.turns[j].src.i == dst.i and
                                type(self.turns[j].dst) == type(src) and self.turns[j].dst.i == src.i and
                                self.turns[j].count == count):
                            # print('Removed turn {}, {}'.format(i, self.turns[i]))
                            # print('Removed turn {}, {}'.format(j, self.turns[j]))
                            del self.turns[j]
                            del self.turns[i]
                            i -= 1
                            break
                        if ((type(self.turns[j].src) == type(dst) and self.turns[j].src.i == dst.i) or
                                (type(self.turns[j].dst) == type(dst) and self.turns[j].dst.i == dst.i) or
                                (type(self.turns[j].src) == type(src) and self.turns[j].src.i == src.i) or
                                (type(self.turns[j].dst) == type(src) and self.turns[j].dst.i == src.i)):
                            break
                    j += 1
            i += 1

        # Repeat pruning until it passes without shortening the solution
        if len(self.turns) < initial_length:
            self.prune()
        else:
            # Rebuild the board list
            self.boards = [self.board]
            for turn in self.turns:
                self.boards.append(turn.apply(self.boards[-1]))

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
                print(i, turn)
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
        b.stacks[1].extend(b.stacks[2].pop())
        self.assertNotEqual(b, b2)

        # Test free space symmetry
        b = Board()
        b.free[0].cards.append(Dragon(SUITS[0]))
        b.free[1].cards.append(Dragon(SUITS[1]))
        b2 = Board()
        b2.free[2].cards.append(Dragon(SUITS[0]))
        b2.free[0].cards.append(Dragon(SUITS[1]))
        self.assertEqual(b, b2)

        # Test stack space symmetry
        b = Board()
        b.stacks[0].cards.append(Dragon(SUITS[0]))
        b.stacks[1].cards.append(Dragon(SUITS[1]))
        b2 = Board()
        b2.stacks[2].cards.append(Dragon(SUITS[0]))
        b2.stacks[0].cards.append(Dragon(SUITS[1]))
        self.assertEqual(b, b2)

    def test_board_hashing(self):
        b = Board()
        b.randomize()
        b2 = Board(b)
        self.assertTrue(len({b, b2}) == 1)
        b.stacks[1].extend(b.stacks[2].pop())
        self.assertTrue(len({b, b2}) == 2)

    def test_solved(self):
        b = Board()
        b.calc_score()
        self.assertTrue(b.is_solved())
        b.randomize()
        b.calc_score()
        self.assertFalse(b.is_solved())
        b = Board()
        b.stacks[4].cards.append(Flower())
        b.calc_score()
        self.assertFalse(b.is_solved())

    def test_string_methods(self):
        """ Test that the board can be parsed to a string and back. """
        b = Board()
        b.randomize()
        b.free[0].extend(b.stacks[2].pop())
        b.free[1].extend([Dragon(SUITS[0])]*4)
        for i in range(1, 7):
            b.goal[1].append(Number(SUITS[1], i))
        b.flower.extend(b.stacks[4].pop())
        b2 = Board()
        b2.from_string(str(b))
        self.assertEqual(b, b2)


if __name__ == '__main__':
    """ Randomly generate boards, then solve them """
    from statistics import mean
    from collections import Counter

    if False:
        seed(47)
        board = Board()
        board.randomize()
        cProfile.run('Solve(board)')
        exit()

    seeds = range(5)
    #seeds = [1, 2, 3, 4, 5, 6, 7, 9]
    #seeds = [1]

    results = Counter()
    turns = []
    pruned = []
    time = []
    boards = []

    for i in seeds:
        # randomize
        seed(i)
        board = Board()
        board.randomize()

        # solve
        solution = Solve(board)

        # prune
        initial_turns = len(solution.turns)
        solution.prune()
        pruned_turns = initial_turns - len(solution.turns)

        # save stats
        results.update([solution.result])
        time.append(solution.duration)
        boards.append(len(solution.board_cache))
        if solution.result == 'solved':
            turns.append(len(solution.turns))
            pruned.append(pruned_turns)

        # print the solution
        #solution.print()

        print('Seed {} {} in {:.3f} seconds after {} boards tested, takes {} moves, {} pruned'.format(
                i, solution.result, solution.duration, len(solution.board_cache), len(solution.turns), pruned_turns))

    # Print the stats
    print()
    for result, count in results.items():
        print('{:12}: {:<4} ({:0.1f}%)'.format(result, count, 100 * count / sum(results.values())))
    try:
        print('Boards seen : min {}, max {}, avg {}'.format(
            min(boards), max(boards), mean(boards)))
        print('Turns       : min {}, max {}, avg {}'.format(
            min(turns), max(turns), mean(turns)))
        print('Turns pruned: min {}, max {}, avg {}'.format(
            min(pruned), max(pruned), mean(pruned)))
        print('Solve time  : min {:0.3f}s, max {:0.3f}s, avg {:0.3f}s'.format(
            min(time), max(time), mean(time)))
    except ValueError:
        print('No Data')
    print()