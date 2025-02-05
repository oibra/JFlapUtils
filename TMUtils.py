import time

class TM():
  def __init__(self, Q, alpha, beta, delta, start, accept, reject):
    self.states = Q
    self.alphabet = alpha
    self.tape = beta
    self.transitions = delta
    self.start = start
    self.accept = accept
    self.reject = reject

  def read(self, input):
    timeout = time.time() + 60
    tape = input
    position = 0
    current = self.start
    configurations = [f"{current}{tape}"]
    while True and time.time() < timeout:
      transition = self.transitions[current][input[position]]
      if transition != None:
        next, write, move = transition
        if next == self.accept:
          return True
        elif next == self.reject:
          return False
        current = next
        if position == len(tape):
          tape += write
        else:
          tape[position] = write
        position += move
        config = f"{tape[:position]}{current}{tape[position:]}"
        if config in configurations:
          return False
        configurations.append(config)
      else: return False