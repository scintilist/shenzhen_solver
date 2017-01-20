# SHENZHEN I/O Solitaire Solver

Automated solver for the solitaire mini-game within SHENZHEN I/O.

#### Performance

Successfully finds solutions to about 95% of games within 10 seconds, with the automated execution taking another 25 seconds on average. Including the time to reset the board and failures, it solves about 100 games per hour. If the 10 second solving timeout is removed, then 99% of games are solvable, but in rare cases may take several minutes to find a solution.

#### Theory
The solving algorithm is weighted A*, with the cost function being the number of turns, and the heuristic function the number of cards remaining in the main stack area. The heuristic function is heavily weighted (ε=4) to massively reduce the solution search time, at the cost of about 5 extra turns over the optimal solutions.

## Getting Started

Launch the Shenzhen I/O Solitaire game in window with at least 1440x900 resolution, then run 'auto_solver.py' to repeatedly solve games.
The auto solver script can be halted by moving the mouse during the execution of the solution.

### Prerequisites

Requires Python 3 with pyautogui and gi

```
>sudo apt-get install python3-gi
>sudo pip3 install pyautogui
```

GUI automation is Linux only, due to the use of Wnck to find the game window.

## Authors

* **Kyle Smith** - *Initial work*

