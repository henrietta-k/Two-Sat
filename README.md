# Two-Sat

Given a list of clauses/restrictions taking the form of (x | ~y) where | denotes "or" and ~ denotes "not", this solver outputs a set of variables that satisfy all the restrictions. This simulates a real-life scenario in which decisions have to be made depending on a set of conditions, such as who to invite to an event. 

## How it Works

The solver coverts the clauses into an implication graph, finds the strongly connected components of this graph, then determines the appropriate variables. It uses Kosaraju's theorem in finding the strongly connected components. 

The output looks something like this:
- a True
- b True
- c False
- d True
