from collections import namedtuple


""" A container for transitions. """
Transition = namedtuple('Transition', ('state', 'action', 'Rt'))
