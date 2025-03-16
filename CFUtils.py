from pyformlang.cfg import Production, Variable, Terminal, CFG
from pyformlang.cfg import Epsilon as cfg_epsilon
from pyformlang.pda import PDA, State, StackSymbol, Symbol
from pyformlang.pda import Epsilon as pda_epsilon
from StringUtils import format_input

class GrammarWrapper():
  def __init__(self, S=None, alpha=set(), V=set(), R=dict()):
    variables = {Variable(v) for v in V}
    terminals = {Terminal(t) for t in alpha}
    start = Variable(S)
    rules = set()
    for v in V:
      for r in R[v]:
        rule = []
        for c in r:
          if c in V:
            rule.append(Variable(c))
          else:
            rule.append(Terminal(c))
        rules.add(Production(Variable(v), rule))
    self.grammar = CFG(variables, terminals, start, rules)


  def read(self, input):
    if input == '':
      return self.grammar.generate_epsilon()
    else:
      s = [Terminal(c) for c in input]
      return self.grammar.contains(s)
  
  def test(self, input, expected=True):
    result = self.read(input)
    if result != expected:
      if expected:
        print(f"reading {format_input(input)} - expected: accept , actual: reject")
      else:
        print(f"reading {format_input(input)} - expected: reject , actual: accept")
    return result == expected

class PDAWrapper():
  def __init__(self, Q=None, alpha=set(), stack=set(), delta={}, q0=None, F=set()):
    s = stack.copy()
    s.add('')
    stack.add('$')

    alpha = alpha.copy()
    alpha.add('')

    self.automata = PDA(states={State(q) for q in Q},
                        start_state=State(q0),
                        start_stack_symbol=StackSymbol('$'),
                        final_states={State(q) for q in F})

    for q in Q:
      for i in alpha:
        j = Symbol(i) if i != '' else pda_epsilon()
        for x in s:
          for r, y in delta[q][i][x]:
            if x == '' and y == '':
              self.automata.add_transitions([(State(q), j, StackSymbol(e), State(r), [StackSymbol(e)]) for e in stack])
            elif x == '':
              self.automata.add_transitions([(State(q), j, StackSymbol(e), State(r), [StackSymbol(y), StackSymbol(e)]) for e in stack])
            elif y == '':
              self.automata.add_transition(State(q), j, StackSymbol(x), State(r), [])
            else:
              self.automata.add_transition(State(q), j, StackSymbol(x), State(r), [StackSymbol(y)])
    
    self.grammar = self.automata.to_empty_stack().to_cfg()


  def read(self, input):
    if input == '':
      return self.grammar.generate_epsilon()
    else:
      return self.grammar.contains(input)
  
  def test(self, input, expected=True):
    result = self.read(input)
    if result != expected:
      if expected:
        print(f"reading {format_input(input)} - expected: accept , actual: reject")
      else:
        print(f"reading {format_input(input)} - expected: reject , actual: accept")
    return result == expected

def parse_variables(productions):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules

  Returns:
    a set of Variables for the given CFG
  """
  var_map = {}
  variables = set()
  for r in productions:
    v = r.find('left').string
    if v:
      var_map[v] = Variable(v)
      variables.add(var_map[v])
  return variables, var_map

def parse_terminals(productions, variables):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules
    variables (set): a set of variables for the given CFG

  Returns:
    a set of Terminals for the given CFG
  """
  # terminals = set()
  term_map = {}
  for r in productions:
    rule = r.find('right').string
    if rule != None:
      for v in variables:
        rule = rule.replace(v.to_text(), '')
      term_map.update({t: Terminal(t) for t in rule})
  return term_map.values(), term_map

def parse_rules(productions, variables, terminals):
  """
  Args:
    productions (list): a list XML nodes representing grammar production rules
    variables (set): a set of variables for the given CFG

  Returns:
    a dict of rules for the given CFG. maps variables to a list of rules, each of which is a list of symbols
  """
  rules = set()
  for r in productions:
    v = variables[r.find('left').string]
    rule = r.find('right').string
    if rule == None:
      rules.add(Production(v, []))
    else:
      rulelist = []
      while rule != '':
        start = [n for n in variables.keys() if rule.startswith(n)]
        if len(start) == 0:
          rulelist.append(terminals[rule[0]])
          rule = rule[1:]
        else:
          var = [a for a in start if len(a) == max([len(n) for n in start])][0]
          rulelist.append(variables[var])
          rule = rule[len(var):]
      rules.add(Production(v, rulelist))
  return rules