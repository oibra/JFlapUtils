from itertools import product
from abc import ABC, abstractmethod
import re # for regex
from CFUtils import CFG, Node

ε = 'ε'

class FA(ABC):
  """
  Superclass for representing a finite automata (DFA or NFA)

  Args:
    Q (set) : set of ids for states in the automata, default: None
    Σ (set) : set of alphabet symbols, default: empty set
    delta (dict) : transition function (w/out ε-transitions). default: {}
      keys are state ids, values are dicts mapping characters to list of resulting states
    e_delta (dict) : dictionary of ε-transitions. default: {}
      keys are state ids, values are lists of resulting states on ε-transitions.
    q0 (str) : id for the start state. default: None
    F (set) : set of all final state ids. default: empty set

  Attributes:
    states (set) : set of ids for states in the automata
    alphabet (set) : set of alphabet symbols
    transitions (dict) : transition function (w/out ε-transitions). 
      keys are state ids, values are dicts mapping characters to list of resulting states
    e_transitions (dict) : dictionary of ε-transitions.
      keys are state ids, values are lists of resulting states on ε-transitions.
    start (str) : id for the start state
    final (set) : set of all final state ids
  """
  def __init__(self, Q=None, Σ=set(), delta={}, e_delta={}, q0=None, F=set()):
    if Q == None:
      Q = {'0'}
      q0 = '0'
      F = set()
      delta = {'0': {c: [] for c in Σ}}
      e_delta = {'0': []}
    self.states = Q
    self.alphabet = Σ
    self.transitions = delta
    self.e_transitions = e_delta
    self.start = q0
    self.final = F 

  def read(self, input):
    """
    Processes given input string using this automata

    Args:
      input (str): input string to read

    Returns:
      Boolean representing whether this automata accepts (True) or rejects (False) the given input
    """
    automata = self.to_dfa()
    curr = automata.start
    for c in input:
      curr = automata.transitions[curr][c][0]
    return curr in automata.final
  
  def test(self, input, expected=True):
    """
    Processes given input string and returns if the acceptance behavior matches expected

    Args:
      input (str) : input string to read
      expected (bool) : expected result of computation

    Returns:
      Boolean representing whether this automata matches the expected behavior on the given input string
    """
    result = self.read(input)
    if result != expected:
      if expected:
        print(f"reading {format_input(input)} - expected: accept , actual: reject")
      else:
        print(f"reading {format_input(input)} - expected: reject , actual: accept")
    return result == expected
    
  def print(self):
    """
    Prints 5-tuple info for this finite automata
    """
    print("states:", self.states)
    print("alphabet:", self.alphabet)
    print("start state:", self.start)
    print("final states:", self.final)
    print("transitions:")
    for s in self.states:
      print("\t",s,":",self.transitions[s],"ε:",self.e_transitions[s])

  def is_empty(self):
    """
    Determines if this automata decides the empty language.

    Returns:
      Boolean representing if this automata accepts the empty langauge. 
      True if it accepts no strings, False otherwise.
    """
    visited = set()
    to_visit = {self.start}
    while len(to_visit) > 0:
      next = to_visit.pop()
      visited.add(next)
      if next in self.final:
        return False
      else:
        to_visit.extend([q for q in self.e_transitions[next] if q not in visited])
        to_visit.extend([q for e in self.alphabet for q in self.transitions[next][e] if q not in visited])
    return True

  def equals(self, other):
    """
    Determines if this automata accepts the same language as the given automata

    Args:
      other (FA): another finite automata

    Returns:
      Boolean representing if the two automata accept the same language.
      True if they are equal, False otherwise.
    """
    this = self.to_dfa()
    that = other.to_dfa()
    # c = this.complement().intersect(that).union(that.complement().intersect(this))
    c = this.minus(that).union(that.minus(this))
    return c.is_empty()
  
  def find_difference(self, other):
    """
    Finds and returns a string that this and the given automata behave differently on, if one exists

    Args:
      other (FA): another finite automata

    Returns:
      string that is either accepted by self and rejected by other, or rejected by self and accepted by other
      None if the two automata decide the same language
    """
    this = self.to_dfa()
    that = other.to_dfa()
    c = this.minus(that).union(that.minus(this))

    visited = set()
    to_visit = {c.start}
    paths = {c.start: ''}
    while len(to_visit) > 0:
      next = to_visit.pop()
      visited.add(next)
      if next in c.final:
        return paths[next]
      else:
        paths.update({q: f"{paths[next]}{e}" for e in c.alphabet for q in c.transitions[next][e] if q not in paths.keys()})
        to_visit.update([q for e in c.alphabet for q in c.transitions[next][e] if q not in visited])
        paths.update({q: paths[next] for q in c.e_transitions[next] if q not in paths.keys()})
        to_visit.update([q for q in c.e_transitions[next] if q not in visited])
    return None


  @abstractmethod
  def to_dfa(self):
    """
    Converts automata to a DFA

    Returns:
      DFA equivalant to this automata
    """
    pass

class DFA(FA):
  """
  Represents a Deterministic Finite Automata.

  Subclass of FA. Must have exactly one transition per state, alphabet pair.
  Must have no epsilon transitions.

  Args:
    Q (set) : set of ids for states in the automata, default: None
    Σ (set) : set of alphabet symbols, default: empty set
    delta (dict) : transition function (w/out ε-transitions). default: {}
      keys are state ids, values are dicts mapping characters to list of resulting states
    e_delta (dict) : dictionary of ε-transitions. default: {}
      keys are state ids, values are lists of resulting states on ε-transitions.
    q0 (str) : id for the start state. default: None
    F (set) : set of all final state ids. default: empty set
  
  Attributes:
    states (set) : set of ids for states in the automata
    alphabet (set) : set of alphabet symbols
    transitions (dict) : transition function (w/out ε-transitions). 
      keys are state ids, values are dicts mapping characters to list of resulting states
    e_transitions (dict) : dictionary of ε-transitions.
      keys are state ids, values are lists of resulting states on ε-transitions.
    start (str) : id for the start state
    final (set) : set of all final state ids

  Raises:
    TypeError: if given 5-tuple does not represent a valid DFA
  """
  def __init__(self, Q=None, Σ=set(), delta={}, e_delta={}, q0=None, F=set()):
    super().__init__(Q, Σ, delta, e_delta, q0, F)
    if not self.is_valid():
      raise TypeError('input not a valid DFA')

  def is_valid(self):
    """
    Determines if automata is a valid DFA.

    Checks that each state, character pair has exactly 1 transition.
    Checks that there are no epsilon transitions.

    Returns:
      True if the automata is a valid DFA, False otherwise.
    """
    for q in self.states:
      for c in self.alphabet:
        if len(self.transitions[q][c]) != 1:
          return False
      if len(self.e_transitions[q]) != 0:
        return False
    return True
  
  def to_dfa(self):
    """See parent class method"""
    return self

  def complement(self):
    """
    Creates a DFA which accepts the complement of the language accepted by this DFA.

    Returns:
      A new DFA whose accepted langauge is the complement of this DFA's accepted langauge.
    """
    return DFA(Q=self.states.copy(), 
            alpha=self.alphabet.copy(),
            delta=self.transitions.copy(),
            e_delta=self.e_transitions.copy(),
            q0=self.start,
            F=self.states.difference(self.final))
  
  def intersect(self, other):
    """
    Creates a DFA which accepts the intersection of this DFA and the given DFA's langauges.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA):

    Returns:
      A new DFA whose accepted langauge is the intersection of this DFA's and the given DFA's accepted lanaguges.
    """
    other = other.to_dfa()
    states = set(product(self.states, other.states))
    return DFA(Q=states,
            alpha=self.alphabet.copy(),
            delta={q: {e: [(self.transitions[q[0]][e][0], other.transitions[q[1]][e][0])] for e in self.alphabet} for q in states},
            e_delta={q: [] for q in states},
            q0=(self.start, other.start),
            F=set(product(self.final, other.final)))
  
  def union(self, other):
    """
    Creates a DFA which accepts the union of this DFA and the given DFA's langauges.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA): 

    Returns:
      A new DFA whose accepted langauge is the union of this DFA's and the given DFA's accepted lanaguges.
    """
    other = other.to_dfa()
    states = set(product(self.states, other.states))
    return DFA(Q=states,
            alpha=self.alphabet.copy(),
            delta={q: {e: [(self.transitions[q[0]][e][0], other.transitions[q[1]][e][0])] for e in self.alphabet} for q in states},
            e_delta={q: [] for q in states},
            q0=(self.start, other.start),
            F={q for q in states if (q[0] in self.final or q[1] in other.final)})
  
  def minus(self, other):
    """
    Creates a DFA which accepts the set difference of this DFA and the given DFA's languages.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA):

    Returns:
      A new DFA whose accepted language is the set difference of this DFA and the given DFA's languages.
    """
    other = other.to_dfa()
    return self.union(other.complement())
      
class NFA(FA):
  """
  Represents a Non-Deterministic Finite Automata.

  Subclass of FA.

  Args:
    Q (set) : set of ids for states in the automata, default: None
    Σ (set) : set of alphabet symbols, default: empty set
    delta (dict) : transition function (w/out ε-transitions). default: {}
      keys are state ids, values are dicts mapping characters to list of resulting states
    e_delta (dict) : dictionary of ε-transitions. default: {}
      keys are state ids, values are lists of resulting states on ε-transitions.
    q0 (str) : id for the start state. default: None
    F (set) : set of all final state ids. default: empty set
  
  Attributes:
    states (set): set of ids for states in the automata
    alphabet (set): set of alphabet symbols
    transitions (dict): transition function. keys are state ids, values are dictionaries that map
      input characters to a list of resulting states.
    start (str): starting state id
    final (set): set of all final state ids
  """
  def __init__(self, Q=None, alpha=set(), delta={}, e_delta={}, q0=None, F=set()):
    super().__init__(Q, alpha, delta, e_delta, q0, F)
    self.dfa = None

  def to_dfa(self):
    """See parent class method"""
    if self.dfa != None:
      return self.dfa
    
    def E(states):  
      Q = states.copy()
      for q in states:
        for s in self.e_transitions[q]:
          if s not in Q:
            Q = Q.union(E({s}))
      return Q

    def multi_is_final(states, final):
      for q in states:
        if q in final:
          return True
      return False

    def get_id(R, ids):
      for i in ids.keys():
        s = ids[i]
        if R <= s and s <= R:
          return i
      return -1

    init = E({self.start})
    ids = {1: init}
    next_id = 2
    final = set()
    if multi_is_final(init, self.final):
      final.add(1)
    transitions = {1: {c: [] for c in self.alphabet}}
    states = {1}
    while len(states) > 0:
      s = states.pop()
      for c in self.alphabet:
        R = set()
        for q in ids[s]:
          R = R.union(set(self.transitions[q][c]))
        R = E(R)
        i = get_id(R, ids)
        if i == -1:
          i = next_id
          ids[i] = R
          states.add(i)
          if multi_is_final(R, self.final):
            final.add(i)
          transitions[i] = {c: [] for c in self.alphabet}
          next_id += 1
        transitions[s][c] = [i]
    states = set(ids.keys())
    
    self.dfa = DFA(Q=states,
               alpha=self.alphabet.copy(),
               delta=transitions,
               e_delta={state: [] for state in states},
               q0=1,
               F=final)
    return self.dfa

  def union(self, other):
    """
    Creates an NFA which accepts the union of this NFA and the given FA's langauges.
    Assumes the alphabets of the two FAs are the same.

    Args:
      other (FA):

    Returns:
      A new NFA whose accepted langauge is the union of this NFA's and the given FA's accepted lanaguges.
    """
    transitions = {0: {x: [] for x in self.alphabet}}
    e_transitions = {0: [f"a{self.start}", f"b{other.start}"]}
    for s in self.states:
      new_s = f"a{s}"
      transitions[new_s] = {x: [] for x in self.alphabet}
      e_transitions[new_s] = []
      for x in self.alphabet:
        for r in self.transitions[s][x]:
          transitions[new_s][x].append(f"a{r}")
      for r in self.e_transitions[new_s]:
        e_transitions[new_s].append(f"a{r}")
    for s in other.states:
      new_s = f"b{s}"
      transitions[new_s] = {x: [] for x in self.alphabet}
      e_transitions[new_s] = []
      for x in self.alphabet:
        for r in other.transitions[s][x]:
          transitions[new_s][x].append(f"b{r}")
      for r in other.e_transitions[new_s]:
        e_transitions[new_s].append(f"b{r}")
    
    return NFA(Q={0}.union({f"a{s}" for s in self.states}).union({f"b{s}" for s in other.states}),
               alpha=self.alphabet.copy(),
               delta=transitions,
               e_delta=e_transitions,
               q0=0,
               F={f"a{s}" for s in self.final}.union({f"b{s}" for s in other.final}))   

  def concat(self, other):
    """
    Creates an NFA which accepts the concatenation of this NFA and the given FA's langauges.
    Assumes the alphabets of the two FAs are the same.

    Args:
      other (FA):

    Returns:
      A new NFA whose accepted langauge is the concatenation of this NFA's and the given FA's accepted lanaguges.
    """
    transitions = {}
    e_transitions = {}
    for s in self.states:
      new_s = f"a{s}"
      transitions[new_s] = {x: [] for x in self.alphabet}
      e_transitions[new_s] = []
      for x in self.alphabet:
        for r in self.transitions[s][x]:
          transitions[new_s][x].append(f"a{r}")
      for r in self.e_transitions[new_s]:
        e_transitions[new_s].append(f"a{r}")
    for s in other.states:
      new_s = f"b{s}"
      transitions[new_s] = {x: [] for x in self.alphabet}
      e_transitions[new_s] = []
      for x in self.alphabet:
        for r in other.transitions[s][x]:
          transitions[new_s][x].append(f"b{r}")
      for r in other.e_transitions[new_s]:
        e_transitions[new_s].append(f"b{r}")
    for s in self.final:
      e_transitions[s].append(f"b{other.start}")
    
    return NFA(Q={f"a{s}" for s in self.states}.union({f"b{s}" for s in other.states}),
               alpha=self.alphabet.copy(),
               delta=transitions,
               e_delta=e_transitions,
               q0=f"a{self.start}",
               F={f"b{s}" for s in other.final})

  def star(self):
    """
    Creates an NFA which accepts the kleene star of this NFA's langauge.

    Returns:
      A new NFA whose accepted langauge is the kleene star of this NFA's lanaguge.
    """
    transitions = {}
    e_transitions = {0: [f"a{self.start}"]}
    for s in self.states:
      new_s = f"a{s}"
      transitions[new_s] = {x: [] for x in self.alphabet}
      e_transitions[new_s] = []
      for x in self.alphabet:
        for r in self.transitions[s][x]:
          transitions[new_s][x].append(f"a{r}")
      for r in self.e_transitions[new_s]:
        e_transitions[new_s].append(f"a{r}")
    for s in self.final:
      e_transitions[s].append('0')
    
    return NFA(Q={f"a{s}" for s in self.states}.union({0}),
               alpha=self.alphabet.copy(),
               delta=transitions,
               e_delta=e_transitions,
               q0=0,
               F={f"a{s}" for s in self.final}.union({0}))

class REGEX:
  """
  Represents a regular expression

  Args:
    string (str): string representation of the regular expression
    alphabet (set): alphabet the regular expression is defined over. default: None

  Attributes:
    string (str): string representation of the regular expression
    alphabet (set): set of characters this regular expression is defined over
    pattern (re): python regex object
    nfa (re): an NFA which decides the language of this regular expression
  """
  def __init__(self, string, alphabet=None):
    self.string = string
    if alphabet == None:
      self.alphabet = guess_alphabet(self.string)
    else:
      self.alphabet = alphabet

    self.pattern = re.compile(self.string)
    self.nfa = None

  def generate_grammar(self):
    """
    Generate a context-free grammar for regular expressions using this expression's alphabet

    Returns:
      A CFG for regular expressions using this expressions alphabet
    """
    start = "<RE>"
    variables = {"<RE>", "<CONCAT>", "<STAR>", "<GROUP>", "<TERM>"}
    terminals = {"", "*", "|", "(", ")"}
    terminals.update(self.alphabet)
    rules = {"<RE>": [["<CONCAT>"], ["CONCAT", "|", "<RE>"]],
             "<CONCAT>": [["<STAR>"], ["<STAR>", "<CONCAT>"]],
             "<STAR>": [["<GROUP>"], ["<GROUP>", "*"]],
             "<GROUP>": [["<TERM>"], ["(", "<RE>", ")"]],
             "<TERM>": [[c] for c in self.alphabet]}

    return CFG(S=start, alpha=terminals, V=variables,R=rules)

  def equals(self, other):    
    """
    Determines if this regular expression is represents the same language as the given regex

    Args:
      other (REGEX): a regular expression

    Returns:
      True if both regular expressions represent the same language
    """
    if isinstance(other, REGEX):
      return self.to_nfa().equals(other.to_nfa())
    else: return False

  def read(self, input):
    """
    Determines if the given input is generated by this regular expression

    Args:
      input (str): an input string
    
    Returns:
      True if the given string is generated by this regular expression, False otherwise
    """
    return self.pattern.fullmatch(input)
  
  def to_nfa(self):
    """
    Creates an NFA which decides the language of this regular expression

    Returns:
      An NFA which decides the language of this REGEX
    """
    if self.nfa != None:
      return self.nfa
    else: 
      grammar = self.generate_grammar()
      parse_tree = grammar.parse_tree(self.string)
      self.nfa = tree_to_nfa(parse_tree, self.alphabet)
      return self.nfa

def tree_to_nfa(node, alphabet):
  """
  Converts the given regular expression AST to an equivalent NFA

  Args:
    node (Node): an AST for a regular expression
    alphabet (set): the alphabet for the NFA to generate

  Returns:
    An NFA equivalent to the given AST
  """
  if node.is_terminal:
    return str_to_nfa(node.string, alphabet)
  else:
    match node.string:
      case "<RE>":
        if len(node.children) == 1:
          return tree_to_nfa(node.children[0], alphabet)
        else:
          left = tree_to_nfa(node.children[0], alphabet)
          right = tree_to_nfa(node.children[2], alphabet)
          return left.union(right)
      case "<CONCAT>":
        if len(node.children) == 1:
          return tree_to_nfa(node.children[0], alphabet)
        else:
          left = tree_to_nfa(node.children[0], alphabet)
          right = tree_to_nfa(node.children[1], alphabet)
          return left.concat(right)
      case "<STAR>":
        if len(node.children) == 1:
          return tree_to_nfa(node.children[0], alphabet)
        else:
          m = tree_to_nfa(node.children[0], alphabet)
          return m.star()
      case "<GROUP>":
        if len(node.children) == 1:
          return tree_to_nfa(node.children[0], alphabet)
        else:
          return tree_to_nfa(node.children[1], alphabet)
      case "<TERM>":
        return tree_to_nfa(node.children[0], alphabet)

def str_to_nfa(string, alphabet):
  """
  Creates an NFA which decides the language containing only the given string

  Args:
    string (str): a given string
    alphabet (set): the alphabet for the language the NFA is defined over
  
  Returns:
    An NFA which decides the language {string} for the given string
  """
  start = 0
  states = set(range(len(string)+1))
  transitions = {state: {c: [] for c in alphabet} for state in states}
  e_transitions = {state: [] for state in states}
  for i in range(len(string)):
    transitions[i][string[i]] = [i+1]
  return NFA(Q=states, alpha=alphabet, delta=transitions, e_delta=e_transitions, q0=start, F=len(string))
  
def format_input(string):
  """
  Reformat the given input string

  Args:
    string (str)
  """
  if len(string) > 0:
    return string
  else:
    return ε

def guess_alphabet(s):
  """
  Attempts to determine the alphabet the given regular expression is defined over.
  """
  alphabet = set()
  for i in range(len(s)):
    if s[i] not in "()*|":
      alphabet.add(s[i])
  return s