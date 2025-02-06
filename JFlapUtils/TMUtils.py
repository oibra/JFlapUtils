import time

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
    tape = input + "□"
    position = 0
    current = self.start
    configurations = [f"{current}{tape}"]
    while True and time.time() < timeout:
      transition = self.transitions[current][input[position]]
      if transition == None:
        return False
      next, write, move = transition
      if next in self.accept:
        return True
      current = next
      if position == len(tape):
        tape += square
      tape[position] = write
      position += move
      config = f"{tape[:position]}{current}{tape[position:]}"
      if config in configurations:
        return False
      configurations.append(config)

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
