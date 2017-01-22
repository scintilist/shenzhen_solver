# SHENZHEN I/O Solitaire Solver

Automated solver for the solitaire mini-game within SHENZHEN I/O.

#### Performance

Successfully finds solutions to about 95% of games within 10 seconds, with the automated execution taking another 25 seconds on average. Including the time to reset the board and failures, it solves about 100 games per hour. If the 10 second solving timeout is removed, then 99% of games are solvable, but in rare cases may take several minutes to find a solution.

#### Theory
The solving algorithm is weighted A*, with the cost function being the number of turns, and the heuristic function the number of cards remaining in the main stack area. The heuristic function is heavily weighted (ε=4) to massively reduce the solution search time, at the cost of about 5 extra turns over the optimal solutions.

## Getting Started

Launch the Shenzhen I/O Solitaire game with at least 1440x900 resolution, then run 'auto_solver.py' to repeatedly solve games.
The auto solver script can be halted by moving the mouse during the execution of the solution.

```
usage: auto_solver.py [-h] [-v] [-d DELAY] [-m MOVESPEED] [-t TIMEOUT]
                      [-w WINCOUNT] [-f] [-s] [-n]

Run the Shenzhen I/O Solitaire auto solver.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verify          verify cards before and after each move, slows
                        execution by ~10%
  -d DELAY, --delay DELAY
                        setup delay in seconds, default=1
  -m MOVESPEED, --movespeed MOVESPEED
                        max mouse move speed in pixels/second, default=50000
  -t TIMEOUT, --timeout TIMEOUT
                        solver timeout in seconds, default=10
  -w WINCOUNT, --wincount WINCOUNT
                        stop after win count reached
  -f, --finish          finish solving the current board, then stop
  -s, --show            show the solution
  -n, --noexec          do not execute the solution

Examples:
>python3 auto_solver.py         Solve and execute as fast as possible indefinitely
>python3 auto_solver.py -fsn    Show the solution to finish the current board.
>python3 auto_solver.py -w 100  Stop when the win count is 100
```

### Prerequisites

Requires Python 3 with pyautogui, gi, and numpy

```
>sudo apt-get install python3-gi
>sudo pip3 install pyautogui
>sudo pip3 install numpy
```

GUI automation is Linux only, due to the use of Wnck to find the game window.

Optional win count tracking can be enabled by installing the module [tesserocr](https://pypi.python.org/pypi/tesserocr).
```
>sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
>sudo pip3 install cython
>sudo pip3 install tesserocr
```
## Authors

* **Kyle Smith** - *Initial work*

