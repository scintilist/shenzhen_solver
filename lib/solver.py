import unittest
from time import perf_counter
import cProfile

from random import shuffle, seed
from collections import namedtuple


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


def card_to_str(card):
    """ Prints the string representation of the card. """
    if isinstance(card, Flower):
        return 'F  '
    if isinstance(card, Dragon):
        return 'D' + card.suit + ' '
    if isinstance(card, Number):
        return 'N' + card.suit + str(card.value)
    raise TypeError


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
    raise ValueError


# Space classes
class Space:
    def __init__(self, space=None):
        if space:
            self.cards = space.cards[:]
            self.key = space.key
            self.hash = space.hash
        else:
            self.cards = []
            self.update()

    def append(self, card):
        self.cards.append(card)
        self.update()

    def extend(self, cards):
        self.cards.extend(cards)
        self.update()

    def pop(self):
        card = self.cards.pop()
        self.update()
        return card

    def update(self):
        self.key = tuple(self.cards)
        self.hash = hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return self.hash


class FlowerSpace(Space):
    pass


class Goal(Space):
    pass


class Free(Space):
    pass


class Main(Space):
    pass


# Board Class
class Board:
    def __init__(self, board=None):
        """ Create the empty board, or copy the board. """
        if board:
            # Cards are only moved, never modified, so it is ok to copy card references, and not duplicate cards
            self.main = [Main(space) for space in board.main]
            self.free = [Free(space) for space in board.free]
            self.goal = [Goal(space) for space in board.goal]
            self.flower = FlowerSpace(board.flower)
        else:
            self.main = [Main() for _ in range(COLUMNS)]
            self.free = [Free() for _ in range(len(SUITS))]
            self.goal = [Goal() for _ in range(len(SUITS))]
            self.flower = FlowerSpace()

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
        for cards in zip(*[iter(deck)] * ROWS):
            space = Main()
            space.extend(cards)
            self.main.append(space)

        # Create the free spaces, goal spaces, and flower space
        self.free = [Free() for _ in range(len(SUITS))]
        self.goal = [Goal() for _ in range(len(SUITS))]
        self.flower = FlowerSpace()

    def next(self):
        """ Generator that yields all possible boards which can be reached with a valid move from the current board. """
        # Main space moves
        for src_i, src in enumerate(self.main):
            # Card stack moves to other main spaces
            for height in range(1, len(src.cards)+1):
                # Break when the stack becomes unmovable
                if height > 1:
                    if not (isinstance(src.cards[-height+1], Number) and isinstance(src.cards[-height], Number) and
                            src.cards[-height+1].suit != src.cards[-height].suit and
                            src.cards[-height+1].value == src.cards[-height].value - 1):
                        break

                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst_i, dst in enumerate(self.main):
                    # Skip moves to self
                    if dst == src:
                        continue

                    # Skip moving the entire column to another empty column
                    if height == len(src.cards) and not dst.cards:
                        continue

                    if dst.cards:
                        # If it can't be stacked on
                        if not (isinstance(src.cards[-height], Number) and isinstance(dst.cards[-1], Number) and
                                src.cards[-height].suit != dst.cards[-1].suit and
                                src.cards[-height].value == dst.cards[-1].value - 1):
                            continue
                    else:
                        # If the destination is empty, but an empty move has already been done
                        if moved_to_empty:
                            continue
                        else:
                            moved_to_empty = True
                    # Copy, move, and yield
                    new_board = Board(self)
                    new_board.main[dst_i].extend(new_board.main[src_i].cards[-height:])
                    new_board.main[src_i].cards = new_board.main[src_i].cards[:-height]
                    new_board.main[src_i].update()
                    yield new_board

            # Single card moves
            if src.cards:
                # To free spaces
                for dst_i, dst in enumerate(self.free):
                    if not dst.cards:
                        # Copy, move, and yield
                        new_board = Board(self)
                        new_board.free[dst_i].append(new_board.main[src_i].pop())
                        yield new_board
                        break

                # To goal spaces
                if isinstance(src.cards[-1], Number):
                    for dst_i, dst in enumerate(self.goal):
                        if src.cards[-1].value == VALUES[0]:
                            if dst.cards:
                                continue
                        else:
                            if not (dst.cards and
                                    src.cards[-1].suit == dst.cards[-1].suit and
                                    src.cards[-1].value == dst.cards[-1].value + 1):
                                continue
                        # Copy, move, and yield
                        new_board = Board(self)
                        new_board.goal[dst_i].append(new_board.main[src_i].pop())
                        yield new_board
                        break

                # To flower space
                if isinstance(src.cards[-1], Flower):
                    # Copy, move, and yield
                    new_board = Board(self)
                    new_board.flower.append(new_board.main[src_i].pop())
                    yield new_board

        # Free space moves
        for src_i, src in enumerate(self.free):
            if len(src.cards) == 1:
                # To goal spaces
                if isinstance(src.cards[-1], Number):
                    for dst_i, dst in enumerate(self.goal):
                        if src.cards[-1].value == VALUES[0]:
                            if dst.cards:
                                continue
                        else:
                            if not (dst.cards and
                                    src.cards[-1].suit == dst.cards[-1].suit and
                                    src.cards[-1].value == dst.cards[-1].value + 1):
                                continue
                        # Copy, move, and yield
                        new_board = Board(self)
                        new_board.goal[dst_i].append(new_board.free[src_i].pop())
                        yield new_board
                        break

                # Move to each main area space
                moved_to_empty = False  # Flag set when moved to an empty stack
                for dst_i, dst in enumerate(self.main):
                    if dst.cards:
                        # If it can't be stacked on
                        if not (isinstance(src.cards[-1], Number) and  isinstance(dst.cards[-1], Number) and
                                src.cards[-1].value == dst.cards[-1].value - 1 and
                                src.cards[-1].suit != dst.cards[-1].suit):
                            continue
                    else:
                        # If the destination is empty, but an empty move has already been done
                        if moved_to_empty:
                            continue
                        else:
                            moved_to_empty = True
                    # Copy, move, and yield
                    new_board = Board(self)
                    new_board.main[dst_i].append(new_board.free[src_i].pop())
                    yield new_board

                # If dragon, try to 'collect' dragons
                if isinstance(src.cards[-1], Dragon):
                    suit = src.cards[-1].suit
                    main_dragons = []
                    for i, space in enumerate(self.main):
                        if space.cards and isinstance(space.cards[-1], Dragon) and space.cards[-1].suit == suit:
                            main_dragons.append(i)
                    free_dragons = []
                    for i, space in enumerate(self.free):
                        if space.cards and isinstance(space.cards[-1], Dragon) and space.cards[-1].suit == suit:
                            free_dragons.append(i)

                    # If the length of the lists of dragon spaces is 4, then pop them all to the free space.
                    if len(free_dragons) + len(main_dragons) == DRAGONS:
                        # Copy, move, and yield
                        new_board = Board(self)
                        for i in free_dragons:
                            new_board.free[src_i].append(new_board.free[i].pop())
                        for i in main_dragons:
                            new_board.free[src_i].append(new_board.main[i].pop())
                        yield new_board

    def is_solved(self):
        """ Returns True if the board is solved. """
        # detects solved boards by checking if the main columns are empty.
        # Assuming all rules were followed, this is only true for a solved board.
        return not any(main.cards for main in self.main)

    def __str__(self):
        """ String representation of the board. """
        s = ''
        # Free spaces, if there is a stack of cards, show the number stacked
        for free in self.free:
            try:
                s += '[' + card_to_str(free.cards[-1])
                s += ']' if len(free.cards) == 1 else str(len(free.cards))
            except IndexError:
                s += '[   ]'

        # Flower space
        try:
            s += '  [' + card_to_str(self.flower.cards[-1]) + ']   '
        except IndexError:
            s += '  [   ]   '

        # Goal spaces
        for goal in self.goal:
            try:
                s += '[' + card_to_str(goal.cards[-1]) + ']'
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
                    s += '[' + card_to_str(main.cards[row]) + ']'
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
            space = Free()
            try:
                index = i * 5 + 1
                card = card_from_string(s[0][index:index+3])
                try:
                    count = int(s[0][index+3])
                except ValueError:
                    count = 1
                space.extend([card] * count)
            except ValueError:
                pass
            self.free.append(space)

        # Load the flower space card
        self.flower = FlowerSpace()
        try:
            index = i * 5 + 8
            self.flower.append(card_from_string(s[0][index:index+3]))
        except ValueError:
            pass

        # Load the goal cards. Generate the full stack down to the 1 card if it is a number
        self.goal = []
        for i in range(len(SUITS)):
            space = Goal()
            try:
                index = len(SUITS) * 5 + i * 5 + 11
                card = card_from_string(s[0][index:index+3])
                if isinstance(card, Number):
                    for value in range(1, card.value+1):
                        space.append(Number(card.suit, value))
                else:
                    space.append(card)
            except ValueError:
                pass
            self.goal.append(space)

        # Load the Main space cards
        self.main = []
        for i in range(COLUMNS):
            space = Main()
            try:
                row = 1
                while True:
                    index = i * 5 + 1
                    card = card_from_string(s[row][index:index+3])
                    space.append(card)
                    row += 1
            except (ValueError, IndexError):
                pass
            self.main.append(space)

    def card_coordinates(self):
        """ Generator that yields all visible cards in the board along with their screen coordinate offsets. """
        # Free spaces
        for i, free in enumerate(self.free):
            try:
                card = free.cards[-1]
            except IndexError:
                pass
            else:
                x = i*152 + 170
                y = 45
                yield card, x, y

        # Flower space
        try:
            card = self.flower.cards[-1]
        except IndexError:
            pass
        else:
            x = 738
            y = 45
            yield card, x, y

        # Goal spaces
        for i, goal in enumerate(self.goal):
            try:
                card = goal.cards[-1]
            except IndexError:
                pass
            else:
                x = i*152 + 930
                y = 45
                yield card, x, y

        # Main spaces
        for i, main in enumerate(self.main):
            row = 0
            while True:
                try:
                    card = main.cards[row]
                except IndexError:
                    break
                else:
                    x = i * 152 + 170
                    y = row*31 + 309
                    yield card, x, y
                row += 1

    def key(self):
        return (frozenset(self.main),) + (frozenset(self.free),) + tuple(self.goal) + (self.flower,)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class Solver:
    def __init__(self):
        self.board_list = []
        self.board_cache = {}
        self.count = 0
        self.timeout = 0

    def solve(self, board):
        """ Solve the puzzle using backtracking. """
        board.next_board = board.next()
        self.board_list = [board]
        self.board_cache = {board}
        self.count = 0

        start_time = perf_counter()
        while not board.is_solved():

            # Generate the next board
            while True:
                try:
                    board = next(self.board_list[-1].next_board)
                    if board not in self.board_cache:
                        board.next_board = board.next()
                        break
                except StopIteration:
                    # If there are no more boards to generate, then pop the board and continue from the previous
                    self.board_list.pop()
                    if not self.board_list:
                        # If there are no more boards to pop, then the board is unsolvable.
                        return 'unsolvable'

            self.board_list.append(board)
            self.board_cache.add(board)
            self.count += 1
            if not self.count % 1000 and self.timeout:
                if perf_counter() - start_time > self.timeout:
                    return 'timed out'
        return 'solved'


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
        b.main[1].append(b.main[2].pop())
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
        b.main[1].append(b.main[2].pop())
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
        b.free[0].append(b.main[2].pop())
        b.free[1].extend([Dragon(SUITS[0])]*4)
        for i in range(1, 7):
            b.goal[1].append(Number(SUITS[1], i))
        b.flower.append(b.main[4].pop())
        b2 = Board()
        b2.from_string(str(b))
        self.assertEqual(b, b2)

if __name__ == '__main__':
    solver = Solver()
    #lib.timeout = 2
    board = Board()

    if False:
        seed(47)
        board.randomize()
        cProfile.run('lib.solve(board)')
        exit()

    longest = 0
    longest_seed = 0
    for i in [1, 2, 3]:
        seed(i)
        board.randomize()
        #print(board)
        start = perf_counter()
        solved = solver.solve(board)
        duration = perf_counter() - start
        if duration > longest:
            longest = duration
            longest_seed = i
        print('Seed {} {} in {:.3f} seconds after {} boards tested'.format(
                i, solved, duration, solver.count))

    print('Longest time to solve was {:.3f} seconds for seed {}'.format(longest, longest_seed))