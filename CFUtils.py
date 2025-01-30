class CFG():
  def __init__(self, S=None, alpha=set(), V=set(), R=dict(), cnf=False):
    self.alphabet = alpha
    self.variables = V
    self.rules = R
    self.start = S

    if cnf:
      self.cnf = self
    else: 
      self.cnf = None

  def to_cnf(self):
    if self.cnf != None:
      return self.cnf
    else: 
      variables = self.variables.copy()
      start = f"{self.start}0"
      variables.add(start)
      rules = {start: [[self.start]]}
      for v in self.variables:
        rules[v] = []
        i = 0
        for rule in self.rules[v]:
          # BIN
          if len(rule) <= 2:
            rules[v].append(rule)
          else:
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
        del_queue = []
        for v in variables:
          if [] in rules[v]:
            del_queue.push(v)
        
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
                  i = rule.index(v)
                  newrule = rule.copy()
                  newrule.pop(i)
                  if newrule != [n] and (n not in deleted or newrule != []):
                    newrules.append(newrule)
                else:
                  if rule[0] != n:
                    newrules.append([rule[0]])
                  if rule[1] != n:
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
            if [v] in rules[v]:
              rules[v].remove(v)
            unit_rules = [r for r in rules[v] if len(r) == 1 and r[0] in variables]

        # TERM
        alpha = ''.join(self.alphabet)
        term = [alpha.index(c) for c in self.alphabet]
        for c in term:
          rules[c] = [[alpha[c]]]
        for v in variables:
          for i in range(len(rules[v])):
            if len(rules[v][i]) == 2:
              if rules[v][i][0] in self.alphabet:
                rules[v][i][0] = alpha.index(rules[v][i][0])
              if rules[v][i][1] in self.alphabet:
                rules[v][i][1] = alpha.index(rules[v][i][1])

      self.cnf = CFG(start, self.alphabet, variables, rules, True)
      return self.cnf

  def check(self, input):
    cnf = self.to_cnf()
    if input == "":
      return [] in cnf.rules[cnf.start]
    else:
      strings = generate(cnf, [cnf.start], len(input))
      return input in strings
  
  def parse_tree(self, input):
    if self.check(input):
      start = Node(self.start)
      self.generate_tree(input, start)
      return start
    else: return None

  def generate_tree(self, input, root):
    if root.string() == input: return True
    elif root.string() != None: return False
    else:
      next = root.first_var_leaf()
      for rule in self.rules[next.string]:
        next.children = [Node(r, r not in self.variables) for r in rule]
        check = self.generate_tree(input, root)
        if check: return True
    return False

class Node():
  def __init__(self, symbol, terminal=False):
    self.string = symbol
    self.is_terminal = terminal
    self.children = []

  def string(self):
    if self.is_terminal: return self.string
    elif len(self.children == 0): return None
    else:
      s = [child.string() for child in self.children]
      if None in s: return None
      else: return ''.join(s)

  def first_var_leaf(self):
    if self.is_terminal: return None
    elif len(self.children == 0): return self
    else:
      for child in self.children:
        n = child.first_var_leaf()
        if n != None: return n
      return None

class PDA():
  def __init__(self, Q=None, alpha=set(), delta={}, e_delta={}, q0=None, F=set()):
    pass

def generate(cfg, phrase=[], length=0):
  V = [var for var in phrase if var in cfg.variables]
  if len(V) == 0 and len(phrase) == length:
    return ''.join(phrase)
  else:
    phrases = set()
    if len(phrase) <= length and len(V) > 0:
      next = V[0]
      for rule in cfg.rules[next]:
        newphrase = replace(phrase, next, rule)
        phrases.update(generate(cfg, newphrase, length))
    return phrases
  
def replace(phrase, var, rule, all=False):
  newphrase = phrase.copy()
  index = phrase.index(var)
  newphrase.pop(index)
  for i in range(len(rule)):
    newphrase.insert(index+i, rule[i])
  return newphrase
