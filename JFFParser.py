from bs4 import BeautifulSoup
from RegularUtils import DFA, NFA, REGEX
from CFUtils import GrammarWrapper, PDAWrapper
from TMUtils import TM, square

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
    transitions = parse_transitions(self.parse_tree, type, states, alphabet)
    return DFA(Q=states, sigma=alphabet, delta=transitions, q0=start, F=final_states)

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
    transitions = parse_transitions(self.parse_tree, type, states, alphabet)
    return NFA(Q=states, sigma=alphabet, delta=transitions, q0=start, F=final_states)

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
    start = productions[0].find('left').string
    rules = parse_rules(productions, variables)

    return GrammarWrapper(start, terminals, variables, rules)
  
  def generate_pda(self):
    type = self.parse_tree.find('type').string
    if type != 'pda':
      raise TypeError('input file is not for a PDA!')
    alphabet = parse_alphabet(self.parse_tree, type)
    stack = parse_stack(self.parse_tree)
    # print(alphabet)
    # print(stack)
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions = parse_transitions(self.parse_tree, type, states, alphabet, stack)
    
    return PDAWrapper(states, alphabet, stack, transitions, start, final_states)
  
  def generate_tm(self):
    type = self.parse_tree.find('type').string
    if type != 'turing':
      raise TypeError('input file is not for a Turing Machine!')
    alphabet = parse_alphabet(self.parse_tree, type)
    states = parse_states(self.parse_tree)
    final_states = parse_final(self.parse_tree)
    start = parse_start(self.parse_tree)
    if start == None:
      raise AttributeError('no start state')
    transitions = parse_transitions(self.parse_tree, type, states, alphabet)
    return TM(states, alphabet, transitions, start, final_states)

### AUTOMATA HELPER FUNCTIONS ###

def parse_alphabet(parse_tree, type):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file
    type (str): the type of JFlap automata

  Returns:
    a set of input characters for the automata
  """
  alphabet = set()
  delta = parse_tree.find_all('transition')
  for e in delta:
    read = e.find('read').string
    if read != None:
      alphabet.add(read)
    if type == 'turing':
      write = e.find('write').string
      if write != None:
        alphabet.add(read)
  return alphabet

def parse_stack(parse_tree):
  alphabet = set()
  delta = parse_tree.find_all('transition')
  for e in delta:
    on = e.find('push').string
    off = e.find('pop').string
    if on != None: alphabet.add(on)
    if off != None: alphabet.add(off)
  return alphabet

def parse_states(parse_tree):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file
  
  Returns:
    a set of states for the automata
  """
  Q = parse_tree.find_all('state')
  return {str(q.get('id')) for q in Q}

def parse_final(parse_tree):
  """
  Args:
    parse_tree: an XML parse tree for a JFlap file

  Returns:
    a set of final states for the automata
  """
  Q = parse_tree.find_all('state')
  return {str(q.get('id')) for q in Q if q.find('final')}

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
  return str(start)

def parse_transitions(parse_tree, type, states, alphabet, stack=None):
  """
  Args:
    parse_tree:
    states (set):
    alphabet (set):

  Returns:
  """
  delta = parse_tree.find_all('transition')
  
  if type == 'fa':
    alphabet = alphabet.union({''})
    transitions = {state: {c: [] for c in alphabet} for state in states}
    # epsilon_transitions = {state: [] for state in states}
    for e in delta:
      q = e.find('from').string
      r = e.find('to').string
      read = '' if e.find('read').string == None else e.find('read').string
      transitions[q][read].append(r)
    return transitions
  elif type == 'pda':
    alphabet = alphabet.union({''})
    stack = stack.union({''})
    transitions = {state: {c: {x: [] for x in stack} for c in alphabet} for state in states}
    # print(transitions)
    for e in delta:
      q = e.find('from').string
      r = e.find('to').string
      read = '' if e.find('read').string == None else e.find('read').string
      push = '' if e.find('push').string == None else e.find('push').string
      pop = '' if e.find('pop').string == None else e.find('pop').string

      if len(push) >= 2:
        raise AttributeError('PDA transitions must push at most 1 character onto the stack')
      if len(pop) >= 2:
        raise AttributeError('PDA transitions must pop at most 1 character off of the stack')

      transitions[q][read][pop].append((r, push))
    return transitions
  elif type == 'turing':
    alphabet = alphabet.union({square})
    transitions = {state: {c: None for c in alphabet} for state in states}
    for e in delta:
      q = e.find('from').string
      r = e.find('to').string
      read = square if e.find('read').string == None else e.find('read').string
      write = square if e.find('write').string == None else e.find('write').string
      move = 1 if e.find('move').string == 'R' else -1
      if transitions[q][read] == None:
        transitions[q][read] = (r, write, move)
      else:
        raise AttributeError(f'TMs must be deterministic; this TM has multiple transitions on {read} from state {q}')
    return transitions

### GRAMMAR HELPER FUNCITONS ###

def parse_variables(productions):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules

  Returns:
    a set of variables for the given CFG
  """
  variables = set()
  for r in productions:
    v = r.find('left').string
    if v:
      variables.add(v)
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
    v = r.find('left').string
    rule = r.find('right').string
    if rule == None:
      rules[v].append([])
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
      rules[v].append(rulelist)
  return rules


if __name__ == "__main__":
  parser = JFFParser('testing/question3b.jff')
  g = parser.generate_tm()
  for s in ['000111222222']:
    print(g.test(s))