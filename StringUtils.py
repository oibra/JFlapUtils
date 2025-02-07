epsilon = 'ε'

def format_input(string):
  """
  Reformat the given input string

  Args:
    string (str)
  """
  return string if len(string) > 0 else epsilon

def guess_alphabet(s, exclude):
  """
  Attempts to determine the alphabet the given regular expression is defined over.
  """
  alphabet = set()
  for i in range(len(s)):
    if s[i] not in exclude:
      alphabet.add(s[i])
  return alphabet

def replace(phrase, var, rule):
  newphrase = phrase.copy()
  index = phrase.index(var)
  newphrase.pop(index)
  for i in range(len(rule)):
    newphrase.insert(index+i, rule[i])
  return newphrase