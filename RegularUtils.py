from itertools import product
from abc import ABC, abstractmethod
from StringUtils import format_input, epsilon, guess_alphabet
from CFUtils import GrammarWrapper
import re


REGEX_ALPHABET = f"()*|{epsilon}"
REGEX_REPLACE = {"+": "|", "!": epsilon}

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
  def __init__(self, Q=None, sigma=set(), delta={}, q0=None, F=set()):
    if Q == None:
      Q = {'0'}
      q0 = '0'
      F = set()
      delta = {'0': {c: [] for c in Σ}}
    self.states = Q
    self.alphabet = sigma
    self.transitions = delta
    self.start = q0
    self.final = F 
    self.sigma_alphabet = self.alphabet.union({''})

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
      print("\t",s,":",self.transitions[s]) 

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
        to_visit.update([q for e in self.sigma_alphabet for q in self.transitions[next][e] if q not in visited])
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
    this = self.to_dfa().minimize()
    that = other.to_dfa().minimize()
    # c = this.minus(that).union(that.minus(this))
    c = this.symmetric_difference(that)
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
    this = self.to_dfa().minimize()
    that = other.to_dfa().minimize()
    # c = this.minus(that).union(that.minus(this))
    c = this.symmetric_difference(that)

    visited = set()
    to_visit = {c.start}
    paths = {c.start: ''}
    while len(to_visit) > 0:
      next = to_visit.pop()
      visited.add(next)
      if next in c.final:
        return paths[next]
      else:
        paths.update({q: f"{paths[next]}{e}" for e in c.sigma_alphabet for q in c.transitions[next][e] if q not in paths.keys()})
        to_visit.update([q for e in c.sigma_alphabet for q in c.transitions[next][e] if q not in visited])
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
  def __init__(self, Q=None, sigma=set(), delta={}, q0=None, F=set(), minimal=False):
    super().__init__(Q, sigma, delta, q0, F)
    self.minimal = minimal
    valid, err = self.is_valid()
    if not valid:
      raise TypeError(f'input not a valid DFA: {err}')

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
          err = f"state {q} has {len(self.transitions[q][c])} transitions on symbol '{c}'"
          print(err)
          return False, err
      if len(self.transitions[q]['']) != 0:
        err = f"state {q} has ε-transitions"
        print(err)
        return False, err
    return True, ''
  
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
            sigma=self.alphabet.copy(),
            delta=self.transitions.copy(),
            q0=self.start,
            F=self.states.difference(self.final),
            minimal=self.minimal)
  
  def intersect(self, other):
    """
    Creates a DFA which accepts the intersection of this DFA and the given DFA's langauges.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA): another finite automata

    Returns:
      A new DFA whose accepted langauge is the intersection of this DFA's and the given DFA's accepted lanaguges.
    """
    other = other.to_dfa()
    temp_states = set(product(self.states, other.states))
    temp_final = set(product(self.final, other.final))
    transitions = {f"{s[0]}x{s[1]}": {c: [f"{self.transitions[s[0]][c][0]}x{other.transitions[s[1]][c][0]}"] for c in self.alphabet} for s in temp_states}
    for s in temp_states:
      transitions[f"{s[0]}x{s[1]}"][''] = []

    result = DFA(Q={f"{s[0]}x{s[1]}" for s in temp_states},
                sigma=self.alphabet.copy(),
                delta=transitions,
                q0=f"{self.start}x{other.start}",
                F={f"{s[0]}x{s[1]}" for s in temp_final})
    return trim_states(result).minimize()
  
  def union(self, other):
    """
    Creates a DFA which accepts the union of this DFA and the given DFA's langauges.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA): another finite automata

    Returns:
      A new DFA whose accepted langauge is the union of this DFA's and the given DFA's accepted lanaguges.
    """
    other = other.to_dfa()
    temp_states = set(product(self.states, other.states))
    transitions = {f"{s[0]}x{s[1]}": {e: [f"{self.transitions[s[0]][e][0]}x{other.transitions[s[1]][e][0]}"] for e in self.alphabet} for s in temp_states}
    for s in temp_states:
      transitions[f"{s[0]}x{s[1]}"][''] = []

    result = DFA(Q={f"{s[0]}x{s[1]}" for s in temp_states},
                sigma=self.alphabet.copy(),
                delta=transitions,
                q0=f"{self.start}x{other.start}",
                F={f"{s[0]}x{s[1]}" for s in temp_states if (s[0] in self.final or s[1] in other.final)})
    return trim_states(result).minimize()
  
  def minus(self, other):
    """
    Creates a DFA which accepts the set difference of this DFA and the given DFA's languages.
    Assumes the alphabets of the two DFAs are the same.

    Args:
      other (FA): another finite automata

    Returns:
      A new DFA whose accepted language is the set difference of this DFA and the given DFA's languages.
    """
    other = other.to_dfa()
    return self.intersect(other.complement())
      
  def symmetric_difference(self, other):
    return self.minus(other).union(other.minus(self))

  def minimize(self):
    """
    Creates a minimal equivalent of this DFA

    Returns:
      A new DFA which is a minimized equivalent of the current DFA
    """
    if self.minimal:
      return self
    
    # helper to find next pair of unmarked states that cannot be merged
    def find_next(merger):
      for q in self.states:
        for r in self.states:
          if merger[q][r] and q != r:
            for c in self.alphabet:
              if not merger[self.transitions[q][c][0]][self.transitions[r][c][0]]:
                return q, r            
      return None, None
    
    # helper to simplify and rename state names
    def simplify(states, transitions, start, final):
      mapping = {}
      i = 0
      for q in states:
        mapping[str(i)] = q
        i += 1
      inverted = {v: k for k,v in mapping.items()}
      new_states = {str(j) for j in range(i)}
      new_transitions = {q: {c: [inverted[transitions[mapping[q]][c][0]]] for c in self.alphabet} for q in new_states}
      for q in new_states:
        new_transitions[q][''] = []

      return DFA(Q=new_states, 
                 sigma=self.alphabet.copy(), 
                 delta=new_transitions, 
                 q0=inverted[start], 
                 F={q for q in new_states if mapping[q] in final}, 
                 minimal=True)
    
    merge = {q: {r: (q in self.final) == (r in self.final) for r in self.states} for q in self.states}    
    next_a, next_b = find_next(merge)
    while next_a != None:
      merge[next_a][next_b] = False
      merge[next_b][next_a] = False
      next_a, next_b = find_next(merge)

    new_states = []
    # old_states = set()
    for q in self.states:
      # m = False
      for r in self.states:
        if r in merge[q].keys() and merge[q][r]:
          # if q != r:
          # m = True
          n = [s for s in new_states if q in s]
          if len(n) > 0:
            new_states[new_states.index(n[0])].add(r)
          else:
            new_states.append({str(q), str(r)})
          del merge[r][q]
      # if not m:
      #   old_states.add(str(q))
    # print(old_states)
    merged_states = set()
    # states = old_states.copy()
    # map_to_new = {q: q for q in old_states}
    map_to_new = {}
    merge_map = {}
    for s in new_states:
      m = ''.join(s)
      merged_states.add(m)
      merge_map[m] = s
      for q in s:
        map_to_new[q] = m
    states = merged_states
    
    # transitions = {q: {c: [map_to_new[self.transitions[q][c][0]]] for c in self.alphabet} for q in old_states}
    transitions = {q: {c: [map_to_new[self.transitions[[r for r in self.states if map_to_new[r] == q][0]][c][0]]] for c in self.alphabet} for q in merged_states}
    for q in states:
      transitions[q][''] = []

    # final = {q for q in old_states if q in self.final}.union({s for s in merged_states if len({q for q in merge_map[s] if q in self.final}) > 0})
    final = {s for s in merged_states if len({q for q in merge_map[s] if q in self.final}) > 0}
    start = map_to_new[self.start]

    return simplify(states, transitions, start, final)

def trim_states(automata):
  visited = set()
  to_visit = {automata.start}
  while len(to_visit) > 0:
    next = to_visit.pop()
    visited.add(next)
    to_visit.update([q for e in automata.sigma_alphabet for q in automata.transitions[next][e] if q not in visited])

  transitions = automata.transitions.copy()
  for q in automata.states:
    if q not in visited:
      del transitions[q]

  return DFA(Q=visited, 
             sigma=automata.alphabet, 
             delta=transitions, 
             q0=automata.start, 
             F={q for q in automata.final if q in visited})

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
  def __init__(self, Q=None, sigma=set(), delta={}, q0=None, F=set()):
    super().__init__(Q, sigma, delta, q0, F)
    self.dfa = None

  def to_dfa(self):
    """See parent class method"""
    if self.dfa != None:
      return self.dfa
    
    def E(states):  
      Q = states.union({r for q in states for r in self.transitions[q]['']})
      if Q != states:
        Q = E(Q)
      return Q

    def multi_is_final(states):
      return len([q for q in states if q in self.final]) > 0

    def get_id(R, ids):
      id = [i for i in ids.keys() if ids[i] == R]
      if len(id) > 0: return id[0]
      return None

    init = E({self.start})
    ids = {'0': init}
    next_id = 1
    transitions = {'0': {c: [] for c in self.alphabet}}
    states_to_finish = {'0'}
    while len(states_to_finish) > 0:
      s = states_to_finish.pop()
      for c in self.alphabet:
        R = E({r for q in ids[s] for r in self.transitions[q][c]})
        i = get_id(R, ids)
        if i == None:
          i = str(next_id)
          ids[i] = R
          states_to_finish.add(i)
          transitions[i] = {c: [] for c in self.alphabet}
          next_id += 1
        transitions[s][c] = [i]
    states = set(ids.keys())
    final = {q for q in states if multi_is_final(ids[q])}

    for q in states:
      transitions[q][''] = []
    
    self.dfa = DFA(Q=states,
               sigma=self.alphabet.copy(),
               delta=transitions,
               q0='0',
               F=final)
    return self.dfa

  def union(self, other):
    """
    Creates an NFA which accepts the union of this NFA and the given FA's langauges.
    Assumes the alphabets of the two FAs are the same.

    Args:
      other (FA): another finite automata

    Returns:
      A new NFA whose accepted langauge is the union of this NFA's and the given FA's accepted lanaguges.
    """
    transitions = {f"a{q}": {c: [f"a{r}" for r in self.transitions[q][c]] for c in self.sigma_alphabet} for q in self.states}
    transitions.update({f"b{q}": {c: [f"b{r}" for r in other.transitions[q][c]] for c in other.sigma_alphabet} for q in other.states})
    transitions['0'] = {c: [] for c in self.alphabet}
    transitions['0'][''] = [f"a{self.start}", f"b{other.start}"]
    
    result = NFA(Q={'0'}.union({f"a{s}" for s in self.states}).union({f"b{s}" for s in other.states}),
               sigma=self.alphabet.copy(),
               delta=transitions,
               q0='0',
               F={f"a{s}" for s in self.final}.union({f"b{s}" for s in other.final}))
    return simplify_nfa(result)   

  def concat(self, other):
    """
    Creates an NFA which accepts the concatenation of this NFA and the given FA's langauges.
    Assumes the alphabets of the two FAs are the same.

    Args:
      other (FA): another finite automata

    Returns:
      A new NFA whose accepted langauge is the concatenation of this NFA's and the given FA's accepted lanaguges.
    """
    transitions = {f"a{q}": {c: [f"a{r}" for r in self.transitions[q][c]] for c in self.sigma_alphabet} for q in self.states}
    transitions.update({f"b{q}": {c: [f"b{r}" for r in other.transitions[q][c]] for c in other.sigma_alphabet} for q in other.states})
    
    for s in self.final:
      transitions[f"a{s}"][''].append(f"b{other.start}")
    
    result = NFA(Q={f"a{s}" for s in self.states}.union({f"b{s}" for s in other.states}),
               sigma=self.alphabet.copy(),
               delta=transitions,
               q0=f"a{self.start}",
               F={f"b{s}" for s in other.final})
    return simplify_nfa(result)

  def star(self):
    """
    Creates an NFA which accepts the kleene star of this NFA's langauge.

    Returns:
      A new NFA whose accepted langauge is the kleene star of this NFA's lanaguge.
    """
    transitions = {f"a{q}": {c: [f"a{r}" for r in self.transitions[q][c]] for c in self.sigma_alphabet} for q in self.states}
    transitions['0'] = {c: [] for c in self.sigma_alphabet}
    transitions['0'][''] = [f"a{self.start}"]
    for s in self.final:
      transitions[f"a{s}"][''].append('0')
    
    return NFA(Q={f"a{s}" for s in self.states}.union({'0'}),
               sigma=self.alphabet.copy(),
               delta=transitions,
               q0='0',
               F={f"a{s}" for s in self.final}.union({'0'}))    

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
    for sym in REGEX_REPLACE.keys():
      self.string = self.string.replace(sym, REGEX_REPLACE[sym])
    if alphabet == None:
      self.alphabet = guess_alphabet(self.string, REGEX_ALPHABET)
    else:
      self.alphabet = alphabet
    self.pattern = re.compile(f"^{self.string.replace(epsilon, '')}$")
    self.nfa = None
    # self.nfa = self.to_nfa()
  
  def __str__(self):
    return self.string
  
  def __repr__(self):
    return self.string

  def generate_grammar(self):
    """
    Generate a context-free grammar for regular expressions using this expression's alphabet

    Returns:
      A CFG for regular expressions using this expressions alphabet
    """
    start = "<RE>"
    variables = {"<RE>", "<CONCAT>", "<STAR>", "<GROUP>", "<TERM>"}
    terminals = {"*", "|", "(", ")", epsilon} | self.alphabet
    term_rules = [[c] for c in self.alphabet]
    group_rules = term_rules + [[epsilon], ["(", "<RE>", ")"]]
    star_rules = term_rules + [[epsilon], ["(", "<RE>", ")"], ["<GROUP>", "*"]]
    concat_rules = term_rules + [[epsilon], ["(", "<RE>", ")"], ["<GROUP>", "*"], ["<STAR>", "<CONCAT>"]]
    union_rules = term_rules + [[epsilon], ["(", "<RE>", ")"], ["<GROUP>", "*"], ["<STAR>", "<CONCAT>"], ["<CONCAT>", "|", "<RE>"]]
    rules = {"<RE>": union_rules,
             "<CONCAT>": concat_rules,
             "<STAR>": star_rules,
             "<GROUP>": group_rules}

    return CFG(S=start, alpha=terminals, V=variables,R=rules)

  def equals(self, other):    
    """
    Determines if this regular expression is represents the same language as the given regex

    Args:
      other (REGEX): another regular expression

    Returns:
      True if both regular expressions represent the same language
    """
    # if isinstance(other, REGEX):
    #   return self.to_nfa().equals(other.to_nfa())
    # else: return False
    return False

  def read(self, input):
    """
    Determines if the given input is generated by this regular expression

    Args:
      input (str): an input string
    
    Returns:
      True if the given string is generated by this regular expression, False otherwise
    """
    return self.pattern.fullmatch(input) != None
  
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
  
  def to_nfa(self):
    """
    Creates an NFA which decides the language of this regular expression

    Returns:
      An NFA which decides the language of this REGEX
    """
    # if self.nfa != None:
    #   return self.nfa
    # else: 
    #   grammar = self.generate_grammar()
    #   parse_tree = grammar.parse_tree(self.string)
    #   self.nfa = tree_to_nfa(parse_tree, self.alphabet)
    #   return self.nfa
    pass

def simplify_nfa(automata):
      mapping = {}
      i = 0
      for q in automata.states:
        mapping[str(i)] = q
        i += 1
      inverted = {v: k for k,v in mapping.items()}
      new_states = {str(j) for j in range(i)}
      new_transitions = {q: {c: [inverted[r] for r in automata.transitions[mapping[q]][c]] for c in automata.sigma_alphabet} for q in new_states}
      return NFA(new_states, 
                 automata.alphabet, 
                 new_transitions, 
                 inverted[automata.start], 
                 {q for q in new_states if mapping[q] in automata.final})

# def tree_to_nfa(node, alphabet):
#   """
#   Converts the given regular expression AST to an equivalent NFA

#   Args:
#     node (Node): an AST for a regular expression
#     alphabet (set): the alphabet for the NFA to generate

#   Returns:
#     An NFA equivalent to the given AST
#   """
#   if node.is_terminal:
#     return str_to_nfa(node.string, alphabet)
#   else:
#     match node.string:
#       case "<RE>":
#         if len(node.children) == 1:
#           return tree_to_nfa(node.children[0], alphabet)
#         elif len(node.children) == 2 and node.children[1] == '*':
#           m = tree_to_nfa(node.children[0], alphabet)
#           return m.star()
#         elif len(node.children) == 2:
#           left = tree_to_nfa(node.children[0], alphabet)
#           right = tree_to_nfa(node.children[1], alphabet)
#           return left.concat(right)
#         elif len(node.children) == 3 and node.children[1] == '|':
#           left = tree_to_nfa(node.children[0], alphabet)
#           right = tree_to_nfa(node.children[2], alphabet)
#           return left.union(right)
#         elif len(node.children) == 3:
#           return tree_to_nfa(node.children[1], alphabet)
#       case "<CONCAT>":
#         if len(node.children) == 1:
#           return tree_to_nfa(node.children[0], alphabet)
#         elif len(node.children) == 2 and node.children[1] == '*':
#           m = tree_to_nfa(node.children[0], alphabet)
#           return m.star()
#         elif len(node.children) == 2:
#           left = tree_to_nfa(node.children[0], alphabet)
#           right = tree_to_nfa(node.children[1], alphabet)
#           return left.concat(right)
#         elif len(node.children) == 3:
#           return tree_to_nfa(node.children[1], alphabet)
#       case "<STAR>":
#         if len(node.children) == 1:
#           return tree_to_nfa(node.children[0], alphabet)
#         elif len(node.children) == 2:
#           m = tree_to_nfa(node.children[0], alphabet)
#           return m.star()
#         elif len(node.children) == 3:
#           return tree_to_nfa(node.children[1], alphabet)
#       case "<GROUP>":
#         if len(node.children) == 1:
#           return tree_to_nfa(node.children[0], alphabet)
#         else:
#           return tree_to_nfa(node.children[1], alphabet)

# def str_to_nfa(string, alphabet):
#   """
#   Creates an NFA which decides the language containing only the given string

#   Args:
#     string (str): a given string
#     alphabet (set): the alphabet for the language the NFA is defined over
  
#   Returns:
#     An NFA which decides the language {string} for the given string
#   """
#   start = '0'
#   if string == epsilon:
#     return NFA(Q={start}, sigma=alphabet, delta={'0': {c: [] for c in alphabet.union({''})}}, q0=start, F={start})
#   else:
#     states = {str(i) for i in range(len(string)+1)}
#     transitions = {state: {c: [] for c in alphabet.union({''})} for state in states}
#     for i in range(len(string)):
#       transitions[str(i)][string[i]] = [str(i+1)]
#     return NFA(Q=states, sigma=alphabet, delta=transitions, q0=start, F={str(len(string))})


  
  

