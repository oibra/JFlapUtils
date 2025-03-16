import time
from StringUtils import format_input

square = '□'

class TM():
  """
  
  Args:
    Q (set)
    alpha (set)
    delta (dict)
    start (str)
    accept (set)

  Attributes:
    states
    alphabet
    transitions
    start
    accept
  """
  def __init__(self, Q, alpha, delta, start, accept):
    self.states = Q
    self.alphabet = alpha
    self.transitions = delta
    self.start = start
    self.accept = accept

  def read(self, input, max_time=60):
    timeout = time.time() + max_time
    tape = square + input + square
    position = 1
    current = self.start
    configurations = [f"{current}{tape}"]
    while time.time() < timeout:
      # print(tape, position)
      # if tape[position] not in self.transitions[current].keys():
      #   return False, True

      transition = self.transitions[current][tape[position]]
      # print(transition)
      if transition == None:
        return False, True
      next, write, move = transition
      if next in self.accept:
        return True, True
      current = next
      if position == len(tape):
        tape += square
      tape = f'{tape[:position]}{write}{tape[position+1:]}'
      # tape[position] = write
      position += move
      config = f"{tape[:position]}{current}{tape[position:]}"
      if config in configurations:
        return False, False
      configurations.append(config)
    return False, False

  def test(self, input, expected=True):
    """
    Processes given input string and returns if the acceptance behavior matches expected

    Args:
      input (str) : input string to read
      expected (bool) : expected result of computation

    Returns:
      Boolean representing whether this automata matches the expected behavior on the given input string
    """
    result, halt = self.read(input)
    if result != expected:
      if expected and halt:
        print(f"reading {format_input(input)} - expected: accept , actual: reject")
      elif expected:
        print(f"reading {format_input(input)} - expected: accept , actual: TM did not halt")
      else:
        print(f"reading {format_input(input)} - expected: reject , actual: accept")
    elif not halt:
      print(f"reading {format_input(input)} - expected: reject , actual: TM did not halt")
    return (result == expected) and halt
