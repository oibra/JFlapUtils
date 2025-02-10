# JFlapUtils

This module (?) creates python utilities for parsing [JFlap](https://www.jflap.org/) files and simulating/using automata, grammars, and regular expressions created in JFlap.
Originally designed for use in autograders at the University of Illinois at Chicago.

Currently supported:
- Finite Automata (DFAs and NFAs)
  - test input strings
  - ~test for equalty~
  - ~find differences between automata~
  - NFA to DFA conversion
  - ~DFA Minimization~
- Regular Expressions
  - test input strings
  - test for equality
  - find differences between automata  
- Context-Free Grammars
  - test if input strings are generated
  - give parse trees for generated strings

In progress:
- Nondeterministic Pushdown Automata
  - test if input strings are generated
- Turing Machines:
  - test if input strings are generated
 
