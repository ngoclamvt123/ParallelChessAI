Parallel Chess AI
=======
This is a Chess AI that has 6 different AI strategies implemented.

Screenshot
==========

    My move: g8f6
    
      8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
      7 ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
      6 · · · · · · · ·
      5 · · · · · · · ·
      4 · · · · · · · ·
      3 · · · · · · · ·
      2 ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙
      1 ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
        a b c d e f g h


    Your move:

Run it!
=======
To use the program, clone this repository. Enter the cloned folder in your terminal, and run the command `python sunfish.py` to use the program with default options. You can also specify options such as the AI strategy, the depth at which to evaluate the minimax tree, and the number of threads to run the strategy if it is one that is parallelized. The strategy can be specified with the flag `--strategy` (or `-s` for short) followed by an integer in the following mapping:

*   1 for Serial Minimax
*   2 for Parallel Top Level Minimax
*   3 for Serial Alpha Beta
*   4 for Parallel Bottom Level Alpha Beta
*   5 for Parallel Top Level Alpha Beta
*   6 for PVSplit

The depth can be specified with the flag `--depth` (or `-d`) followed by a positive integer. The number of threads can be specified with the flag `--threads` (or `-t`) followed by a positive integer. By default, we have the program run the Serial Minimax at a depth of 3 with, of course, 1 thread. 

If we wanted to run the program with, for example, the Parallel Top Level Alpha Beta strategy at a depth of 5 with 4 threads, we would run the following command: `python sunfish.py --strategy 5 --depth 5 --threads 4` or `python sunfish.py -s 5 -d 5 -t 4` for short. 

You can also view the options and their descriptions by running `python sunfish.py -h`. Running this command will return the following descriptions: 

![Screen Shot 2015-12-09 at 8.37.08 PM](https://parallelchess.files.wordpress.com/2015/12/screen-shot-2015-12-09-at-8-37-08-pm.png) 

Once you get the program running, a board will be returned. You (the human) are playing against the computer, and will enter the first move. 

![Screen Shot 2015-12-09 at 8.51.14 PM](https://parallelchess.files.wordpress.com/2015/12/screen-shot-2015-12-09-at-8-51-14-pm.png) 

The move will take the format `e2e4`, where `e4` is the place you want to move the piece at `e2` to. Once you enter a valid move, the program will return a new board with the move you entered applied: 

![Screen Shot 2015-12-09 at 9.08.50 PM](https://parallelchess.files.wordpress.com/2015/12/screen-shot-2015-12-09-at-9-08-50-pm.png) 

The computer will then respond with its move, as well as information on the response, such as the number of board states it explored and the time it took to find an optimal move: 

![Screen Shot 2015-12-09 at 9.02.10 PM](https://parallelchess.files.wordpress.com/2015/12/screen-shot-2015-12-09-at-9-02-10-pm.png) 

You can continue playing the computer until there is a winner. Good luck!
