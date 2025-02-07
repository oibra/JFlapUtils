from StringUtils import format_input, replace

class CFG():
  """
  A class for representing a context-free grammar

  Args:
    S (str):
    alpha (set):
    V (set):
    R (dict):

  Attributes:
    alphabet (set)
    varibales (set)
    rules (dict)
    start (str)
  """
  def __init__(self, S=None, alpha=set(), V=set(), R=dict(), cnf=False):
    self.alphabet = alpha
    self.variables = V
    self.rules = R
    self.start = S

    if cnf:
      self.cnf = self
    else: 
      self.cnf = None

  class Node():
    """"""
    def __init__(self, symbol, terminal=False):
      self.string = symbol
      self.is_terminal = terminal
      self.children = list()

    def __str__(self):
      return f"({self.string}, {[str(child) for child in self.children]})"
    
    def __repr__(self):
      return f"({self.string}, {[child for child in self.children]})"

    def to_string(self):
      """"""
      if self.is_terminal: return self.string
      elif len(self.children) == 0: return None
      else:
        s = [child.to_string() for child in self.children]
        if None in s: return None
        else: return ''.join(s)

    def find_unexplored(self, v):
      if self.is_terminal: return None
      elif len(self.children) == 0 and self.string == v: return self
      else:
        for child in self.children:
          n = child.find_unexplored(v)
          if n != None: return n
        return None

    def first_var_leaf(self):
      """"""
      if self.is_terminal: return None
      elif len(self.children) == 0: return self
      else:
        for child in self.children:
          n = child.first_var_leaf()
          if n != None: return n
        return None
    
    def copy(self):
      copy = CFG.Node(self.string, self.is_terminal)
      copy.children = [child.copy() for child in self.children]
      return copy

  def to_cnf(self):
    """
    Creates an equivlant CFG in Chomsky-Normal Form (CNF) and returns it

    Returns:
    """
    if self.cnf != None:
      return self.cnf
    variables = self.variables.copy()
    start = f"{self.start}0"
    variables.add(start)
    rules = {start: [[self.start]]}
    # BIN
    for v in self.variables:
      rules[v] = [rule for rule in self.rules[v] if len(rule) <= 2]
      i = 0
      for rule in self.rules[v]:
        if len(rule) > 2:
          i += 1
          newv = f"{v}{i}"
          rules[v].append([rule[0], newv])
          variables.add(newv)
          newrule = rule[1:]
          while len(newrule) > 2:
            oldv = newv
            i += 1
            newv = f"{v}{i}"
            rules[oldv] = [[newrule[0], newv]]
            newrule = newrule[1:]
          rules[newv] = [newrule]
    # DEL
    del_queue = [v for v in variables if [] in rules[v]]
    deleted = []
    while len(del_queue) > 0:
      v = del_queue.pop()
      deleted.push(v)
      rules[v].remove([])
      for n in variables:
        newrules = []
        for rule in rules[n]:
          if v in rule:
            if rule.count(v) == 1:
              newrule = rule.copy()
              newrule.pop(rule.index(v))
              if newrule != [n] and (n not in deleted or newrule != []):
                newrules.append(newrule)
            else:
              if v != n:
                newrules.append([rule[0]])
                newrules.append([rule[1]])
              if n not in deleted:
                newrules.append([])
        if [] in newrules and n not in del_queue and n != start:
          del_queue.push(n)
        rules[n].extend(newrules)
    # UNIT
    for v in variables:
      unit_rules = [r for r in rules[v] if len(r) == 1 and r[0] in variables]
      while(len(unit_rules) > 0):
        unit = unit_rules.pop()
        rules[v].remove(unit)
        rules[v].extend(rules[unit[0]])
        if [v] in rules[v]: rules[v].remove([v])
        unit_rules = [r for r in rules[v] if len(r) == 1 and r[0] in variables]
    # TERM
    alpha = ''.join(self.alphabet)
    term = [alpha.index(c) for c in self.alphabet]
    rules.update({f"V{c}": [alpha[c]] for c in term})
    for v in variables:
      for rule in rules[v]:
        if len(rule) == 2:
          if rule[0] in self.alphabet:
            rule[0] = f"V{alpha.index(rule[0])}"
          if rule[1] in self.alphabet:
            rule[1] = f"V{alpha.index(rule[1])}"
    variables.update(f"V{c}" for c in term)

    self.cnf = CFG(start, self.alphabet, variables, rules, True)
    return self.cnf

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

  def read(self, input):
    """
    Determines if the given input is generated by this grammar

    Args:
      input (str)

    Returns:
    """
    cnf = self.to_cnf()
    if input == "": return [] in cnf.rules[cnf.start]
    else: 
      find, _ = cnf.generate(self.Node(cnf.start), input, [cnf.start])
      return find
  
  def parse_tree(self, input):
    """
    Generates a parse tree for the given string in the cnf form of this grammar, if it is generated

    Args:
      input (str)

    Returns:
    """
    if input != "":
      find, tree = self.generate(input, self.Node(self.start), [self.start])
      if find: return tree
    return None
  
  def generate(self, goal, root, phrase):
    V = None
    for sym in phrase:
      if sym in self.variables:
        V = sym
        break

    if V == None and len(phrase) == len(goal):
      p = ''.join(phrase)
      return (p == goal), root.copy()
    elif len(phrase) <= len(goal) and V != None:
      curr = root.find_unexplored(V)
      for rule in self.rules[V]:
        if len(rule) + len(phrase) - 1 <= len(goal):
          terms = [c for c in rule if c not in self.variables and phrase.count(c) >= goal.count(c)]
          if len(terms) == 0:
            newphrase = replace(phrase, V, rule)
            curr.children = [CFG.Node(child, child not in self.variables) for child in rule]
            find, tree = self.generate(goal, root, newphrase)
            if find:  
              return find, tree
            curr.children = []
    return False, root

class PDA():
  """"""
  def __init__(self, Q=None, alpha=set(), stack=set(), delta={}, q0=None, F=set()):
    self.states = Q
    self.sigma = alpha
    self.gamma = stack
    self.transitions = delta
    self.start = q0 
    self.final = F

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

  def read(self, input, curr=None, stack=[]):
    if curr == None: curr = self.start
    if input == '' and curr in self.final:
      return True
    for q,s in self.transitions[curr]['']['']:
      if s != '': stack.append(s)
      if self.read(input, q, stack): return True
      if s != '': stack.pop()
      
    x = stack[-1]
    stack.pop()
    for q,s in self.transitions[curr][''][x]:
      if s != '': stack.append(s)
      if self.read(input, q, stack): return True
      if s != '': stack.pop()
    stack.append(x)

    if input != '':
      for q,s in self.transitions[curr][input[0]]['']:
        if s != '': stack.append(s)
        if self.read(input[1:], q, stack): return True
        if s != '': stack.pop()
      x = stack[-1]
      stack.pop()
      for q,s in self.transitions[curr][input[0]][x]:
        if s != '': stack.append(s)
        if self.read(input[1:], q, stack): return True
        if s != '': stack.pop()
      stack.append(x)
    return False
