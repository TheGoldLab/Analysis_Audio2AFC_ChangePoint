"""
Python module to analyze mental model comlexity in our Auditory change-point task
"""
import numpy as np

SIDES = {'left', 'right'}


def check_valid_side(side):
    """
    Check that side is in the allowed set of sides

    Args:
        side: (str) usually either 'left' or 'right'

    Returns: None if side is valid. Otherwise raises ValueError

    """
    if side not in SIDES:
        raise ValueError(f"{side} is not a valid side")
    return None


def switch_side(side):
    """

    Args:
        side: (str) either 'left' or 'right'

    Returns: The opposite side

    """

    check_valid_side(side)

    if len(SIDES) != 2:
        raise RuntimeError(f"This function shouldn't be used with len(SIDES)={len(SIDES)}")

    opposite_side = next(iter(SIDES - {side}))
    return opposite_side


class Stimulus:
    """Define stimulus object, which is a sequence of consecutive trials"""

    likelihood_same_side = 0.8
    """Likelihood of a sound occurring on the same side as the source"""

    def _generate_sound_from_source(self, source):
        """
        Generates a random sound location for a given source

        Args:
            source: (str) either 'left' or 'right'

        Returns: (str) either 'left' or 'right'

        """
        check_valid_side(source)

        same_side = np.random.binomial(1, self.likelihood_same_side)

        return source if same_side else switch_side(source)
    

class BinaryDecisionMaker:
    """Simulate an observer performing our Auditory change-point 2AFC task"""
    def observe(self, stimulus):
        pass
    def process(self):
        pass
    def decide(self):
        pass

