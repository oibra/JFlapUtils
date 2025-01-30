from bs4 import BeautifulSoup
from RegularUtils import DFA, NFA, REGEX

class JFFParser():
  def __init__(self, filename):
    with open(filename, 'r') as file:
      data = file.read()
    bs_data = BeautifulSoup(data, 'xml')
    self.parse_tree = bs_data

  def generate_dfa(self):
    type = self.parse_tree.find('type').string
    if type != "fa":
      raise AttributeError('input file is not for a finite autoamta!')

    alphabet = parse_alphabet(self.parse_tree, "FA")
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions, epsilon_transitions = parse_transitions(self.parse_tree, states, alphabet)
    return DFA(Q=states, alpha=alphabet, delta=transitions, e_delta=epsilon_transitions, q0=start, F=final_states)

  def generate_nfa(self):
    type = self.parse_tree.find('type').string
    if type != "fa":
      raise AttributeError('input file is not for a finite autoamta!')

    alphabet = parse_alphabet(self.parse_tree, "FA")
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions, epsilon_transitions = parse_transitions(self.parse_tree, states, alphabet)
    
    return NFA(Q=states, alpha=alphabet, delta=transitions, e_delta=epsilon_transitions, q0=start, F=final_states)

  def generate_regex(self):
    type = bs_data.find('type').string
    if type != "re":
      raise AttributeError('input file is not for a regular expression!')
    
    string = bs_data.find('expression').string
    return REGEX(string)
    

def parse_alphabet(parse_tree, type):
  if type == "FA":
    alphabet = set()
    delta = parse_tree.find_all('transition')
    for e in delta:
      read = e.find('read').string
      if read != None:
        alphabet.add(read)
    return alphabet

def parse_states(parse_tree):
  Q = parse_tree.find_all('state')
  return {q.get('id') for q in Q}

def parse_final(parse_tree):
  Q = parse_tree.find_all('state')
  return {q.get('id') for q in Q if q.find('final')}

def parse_start(parse_tree):
  Q = parse_tree.find_all('state')
  start = None
  for q in Q:
    if q.find('initial'):
      if start == None:
        start = q.get('id')
      else:
        raise AttributeError('too many start states')
  return start

def parse_transitions(parse_tree, states, alphabet):
  delta = parse_tree.find_all('transition')
  transitions = {state: {c: [] for c in alphabet} for state in states}
  epsilon_transitions = {state: [] for state in states}
  for e in delta:
    read = e.find('read').string
    if read != None: 
      transitions[e.find('from').string][read].append(e.find('to').string)
    else: 
      epsilon_transitions[e.find('from').string].append(e.find('to').string)
  return transitions, epsilon_transitions
