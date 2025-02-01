from bs4 import BeautifulSoup
from RegularUtils import DFA, NFA, REGEX
from CFUtils import CFG

class JFFParser():
  """
  Args:
    filename (str)

  Attributes:
    parse_tree: parse tree for the jflap xml
  """
  def __init__(self, filename):
    with open(filename, 'r') as file:
      data = file.read()
    bs_data = BeautifulSoup(data, 'xml')
    self.parse_tree = bs_data

  def generate_dfa(self):
    """
    Returns:
      A DFA generated from the JFlap file
    """
    type = self.parse_tree.find('type').string
    if type != "fa":
      raise AttributeError('input file is not for a finite autoamta!')

    alphabet = parse_alphabet(self.parse_tree, type)
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions, epsilon_transitions = parse_transitions(self.parse_tree, states, alphabet)
    return DFA(Q=states, alpha=alphabet, delta=transitions, e_delta=epsilon_transitions, q0=start, F=final_states)

  def generate_nfa(self):
    """
    Returns:
      An NFA generated from the JFlap file
    """
    type = self.parse_tree.find('type').string
    if type != "fa":
      raise AttributeError('input file is not for a finite autoamta!')

    alphabet = parse_alphabet(self.parse_tree, type)
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions, epsilon_transitions = parse_transitions(self.parse_tree, states, alphabet)
    return NFA(Q=states, alpha=alphabet, delta=transitions, e_delta=epsilon_transitions, q0=start, F=final_states)

  def generate_regex(self):
    """
    Returns:
      A REGEX generated from the JFlap file
    """
    type = self.parse_tree.find('type').string
    if type != "re":
      raise AttributeError('input file is not for a regular expression!')
    
    string = self.parse_tree.find('expression').string
    return REGEX(string)
    
  def generate_grammar(self):
    """
    Returns:
      A CFG generated from the JFlap file
    """
    type = self.parse_tree.find('type').string
    if type != 'grammar':
      raise TypeError('input file is not for a grammar!')
    productions = self.parse_tree.find_all('production')
    variables = parse_variables(productions)
    terminals = parse_terminals(productions, variables)
    start = productions[0].find('right').string
    rules = parse_rules(productions, variables)

    return CFG(start, terminals, variables, rules)


def parse_variables(productions):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules

  Returns:
    a set of variables for the given CFG
  """
  variables = set()
  for r in productions:
    variables.add(r.find('left').string)
  return variables

def parse_terminals(productions, variables):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules
    variables (set): a set of variables for the given CFG

  Returns:
    a set of terminals for the given CFG
  """
  terminals = set()
  for r in productions:
    rule = r.find('right').string
    if rule != None:
      for v in variables:
        rule = rule.replace(v, '')
      terminals.update(list(rule))
  return terminals

def parse_rules(productions, variables):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules
    variables (set): a set of variables for the given CFG

  Returns:
    a dict of rules for the given CFG. maps variables to a list of rules, each of which is a list of symbols
  """
  rules = {v: [] for v in variables}
  for r in productions:
    v = r.find('right').string
    rule = r.find('left').string
    if rule == None:
      rules[v].push([])
    else:
      rulelist = []
      while rule != '':
        start = [n for n in variables if rule.startswith(n)]
        if len(start) == 0:
          rulelist.append(rule[0])
          rule = rule[1:]
        else:
          var = [a for a in start if len(a) == max([len(n) for n in start])][0]
          rulelist.append(var)
          rule = rule[len(var):]
      rules[v].push(rulelist)


def parse_alphabet(parse_tree, type):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file
    type (str): the type of JFlap automata

  Returns:
    a set of input characters for the automata
  """
  if type == "fa":
    alphabet = set()
    delta = parse_tree.find_all('transition')
    for e in delta:
      read = e.find('read').string
      if read != None:
        alphabet.add(read)
    return alphabet

def parse_states(parse_tree):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file
  
  Returns:
    a set of states for the automata
  """
  Q = parse_tree.find_all('state')
  return {q.get('id') for q in Q}

def parse_final(parse_tree):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file

  Returns:
    a set of final states for the automata
  """
  Q = parse_tree.find_all('state')
  return {q.get('id') for q in Q if q.find('final')}

def parse_start(parse_tree):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file

  Returns:
    the start state for the automata
  """
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
  """
  Args:
    parse_tree:
    states (set):
    alphabet (set):

  Returns:
  """
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
