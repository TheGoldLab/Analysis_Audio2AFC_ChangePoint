"""
Python module to analyze mental model comlexity in our Auditory change-point task
"""
import numpy as np
from scipy.stats import bernoulli

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

    def __init__(self, num_trials, hazard, sources=None, sounds=None):
        self.num_trials = num_trials

        if isinstance(hazard, float):
            if 0 <= hazard <= 1:
                self.hazard = hazard
            else:
                raise ValueError(f"hazard rate should be between 0 and 1")
        else:
            raise ValueError(f"Right now, only scalar hazard rate between 0 and 1 are accepted")

        self.source_sequence = self.generate_source_sequence()

    def _generate_sound_from_source(self, source):
        """
        Generates a random sound location for a given source

        Args:
            source: (str) either 'left' or 'right'

        Returns: (str) either 'left' or 'right'

        """
        check_valid_side(source)

        same_side = bernoulli.rvs(self.likelihood_same_side)

        return source if same_side else switch_side(source)

    def generate_source_sequence(self, init, hazard):
        pass


class BinaryDecisionMaker:
    """Simulate an observer performing our Auditory change-point 2AFC task"""

    mislocalization_noise = 0
    """probability with which the observer hears a tone on the wrong side"""

    bias = 0.5
    """probability with which observer picks 'right' when guessing. Unbiased corresponds to 0.5"""

    likelihoods_known = True
    """if true, model uses true probability that the sound occurs on same side as source"""
    if not likelihoods_known:
        raise NotImplementedError

    sources_prior = {'left':.5, 'right':.5}
    """prior expectations about most likely side of a source"""

    def __init__(self, stimulus_object):
        """
        Args:
            stimulus_object: a stimulus object (instance of Stimulus)
        """
        self.stimulus_object = stimulus_object

    def observe(self, list_of_sounds):
        """
        Generate subjective observations of a given stimulus

        todo: not clear yet whether list_of_sounds should be automatically extracted from the stimulus_object attribute

        Args:
            list_of_sounds: list of sound locations (str)

        Returns: list of perceived sound locations (which we call observations)

        """
        # check all stimulus values are valid
        _ = map(check_valid_side, list_of_sounds)  # exception raised if a stimulus is invalid

        def apply_sensory_noise(side):
            """
            Intermediate function that switches sound location according to sensory noise

            Args:
                side: (str) a member of global set SIDES

            Returns: perceived side after sensory noise is applied

            """
            return switch_side(side) if bernoulli.rvs(self.mislocalization_noise) else side

        return map(apply_sensory_noise, list_of_sounds)

    def process(self, observations, hazard=None):
        """
        This is where the bulk of the decision process occurs. Observations are converted into a decision variable.

        For now, only the log posterior odds of the sources is computed, and hazard rate is assumed fixed.

        Args:
            observations: sequence of perceived sound locations (list)
            hazard: hazard rate, if None, the one from the stimulus_object attribute is fetched

        Returns: generator object for decisions

        """
        # check all observations values are valid
        _ = map(check_valid_side, observations)  # exception raised if a stimulus is invalid

        if hazard is None:
            hazard = self.stimulus_object.hazard
        assert isinstance(hazard, float) and (0 <= hazard <= 1)

        prob_same_side = self.stimulus_object.likelihood_same_side

        # jump in accrued evidence towards 'right' choice if sound on right
        jump_magnitude = np.log(prob_same_side / (1 - prob_same_side))

        def discount_old_evidence(y):
            """
            Discount evidence from last time point in optimal sequential decision making in changing environment

            hazard rate is assumed known

            Args:
                y: evidence (log posterior odds) of previous time step

            Returns: float - positive favors 'right', negative 'left'

            """
            numerator = hazard * np.exp(-y) + 1 - hazard
            denominator = hazard * np.exp(y) + 1 - hazard
            return np.log(numerator / denominator)

        def recursive_update():
            prior_belief = np.log(self.sources_prior['right'] / self.sources_prior['left'])
            num_observations = len(observations)
            decision_number = 0
            while decision_number < num_observations:
                if observations[decision_number] == 'right':
                    jump = jump_magnitude
                else:
                    jump = -jump_magnitude

                if not decision_number:  # if this is the first decision
                    log_posterior_odds = prior_belief

                log_posterior_odds += jump + discount_old_evidence(log_posterior_odds)

                yield self.decide(log_posterior_odds)
                decision_number += 1

        return recursive_update

    def decide(self, decision_variable):
        """
        Makes a decision on a single trial, based on the decision variable

        Args:
            decision_variable: for now, log posterior odds

        Returns: (str) an element from SIDES

        """
        s = np.sign(decision_variable)
        if s == -1:
            return 'left'
        elif s == 1:
            return 'right'
        elif s == 0:
            return 1 if bernoulli.rvs(self.bias) else 0
