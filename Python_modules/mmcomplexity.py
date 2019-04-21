"""
Python module to analyze mental model comlexity in our Auditory change-point task
"""
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, beta

SIDES = {'left', 'right'}
"""set of allowed sides"""


def flag_change_points(seq):
    """
    iterate through seq and flag change points with False boolean.

    Args:
        seq (list or ndarray): array-like object. Must be iterable and each element in iteration must support '=='
            list and ndarray types are not enforced

    Raises:
        ValueError: if seq is empty, or if one of its elements is not 1-D

    Returns:
        generator: generates boolean values, True whenever a term in seq differs from its predecessor

    """
    def check_1d(i):
        """
        This function raises exceptions when i is either a list or an ndarray with a different number of elements than 1

        Args:
            i: anything
        Raises:
            ValueError: if i is either a list or an ndarray with a different number of elements than 1
        Returns:
            None

        """
        if isinstance(i, np.ndarray) and i.size != 1:
            raise ValueError('rows of ndarray have more than 1 element')
        elif isinstance(i, list) and len(i) != 1:
            raise ValueError('an element from array has more or less than 1 element')

    new_seq = list(seq)
    if new_seq:  # checks that seq was not empty
        last_item = new_seq[0]
        check_1d(last_item)

        yield False  # first time step is never a change-point

        for next_item in new_seq[1:]:
            check_1d(next_item)
            yield next_item != last_item
            last_item = next_item
    else:
        raise ValueError('provided iterable is empty')


def check_valid_side(side):
    """
    Check that side is in the allowed set of sides

    Args:
        side (str): usually either 'left' or 'right'
    Raises:
        ValueError: if side is hashable but invalid
        TypeError: if side is not hashable
    Returns:
        None: if side is valid.
    """
    if side not in SIDES:
        raise ValueError(f"{side} is not a valid side")
    return None


def check_valid_sequence_of_sides(sides):
    """
    Check that all elements in sides are valid sides

    Args:
        sides (list): list of sides

    Returns:

    """
    assert isinstance(sides, list)
    _ = map(check_valid_side, sides)


def switch_side(side):
    """

    Args:
        side (str):  an element of SIDES representing the side we want to switch from
    Raises:
        RunTimeError: if len(SIDES) != 2
    Returns:
        str: The opposite side

    """

    check_valid_side(side)

    if len(SIDES) != 2:
        raise RuntimeError(f"This function shouldn't be used with len(SIDES)={len(SIDES)}")

    opposite_side = next(iter(SIDES - {side}))
    return opposite_side


def get_next_change_point(p):
    """
    Sample from geometric distribution to tell us when the next change point will occur

    See `doc <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.geometric.html>`_ if unclear

    Args:
        p: Bernoulli parameter between 0 and 1

    Returns:
        int: time step in the future for occurrence of first success (starts counting at 1)
    """
    return np.random.geometric(p)


def infer_bernoulli_bayes(num_successes, num_trials, beta_prior=(1, 1)):
    """
    Given ``num_trials`` independent observations from a Bernoulli random variable with ``num_successes`` successes,
    returns the posterior distribution over the Bernoulli parameter in the form of a Beta distribution. May
    take hyperparameters of a Beta distribution for the prior.

    To compute the posterior, the sufficient statistics are updated.

    Args:
        num_successes (int): number of successes
        num_trials (int): number of observations
        beta_prior (tuple): corresponds to the usual parameters of a Beta distribution, a and b (or alpha, beta)
            defaults to (1,1), which is a flat prior
    Raises:
        ValueError: if num_trials < num_successes or a hyperparameter is negative or num_trials < 0

    Returns:
        scipy.stats.beta: frozen distribution, see
            `doc <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.beta.html>`_.

    """
    if num_trials < 0:
        raise ValueError('negative number of trials')
    if num_trials < num_successes:
        raise ValueError('fewer trials than sucesses')
    if beta_prior[0] < 0 or beta_prior[1] < 0:
        raise ValueError('hyperprior cannot have negative parameters')

    return beta(beta_prior[0] + num_successes, beta_prior[1] + num_trials - num_successes)


class StimulusBlock:
    """Define stimulus for a block of trials in which hazard rate is fixed"""

    source_prior = {'left': 0.5, 'right': 0.5}

    likelihood_same_side = 0.8
    """Likelihood of a sound occurring on the same side as the source"""

    def __init__(self, num_trials, hazard, first_source=None, sources=None, sounds=None):
        self.num_trials = num_trials

        if isinstance(hazard, float) or isinstance(hazard, int):
            if 0 <= hazard <= 1:
                self.hazard = hazard
            else:
                raise ValueError(f"hazard rate should be between 0 and 1")
        else:
            raise ValueError(f"Right now, only scalara float or int hazard rate between 0 and 1 are accepted")

        if sources is None:
            self.source_sequence = self.generate_source_sequence(first_source)
        else:
            check_valid_sequence_of_sides(sources)
            self.source_sequence = sources

        if sounds is None:
            self.sound_sequence = list(map(self._generate_sound_from_source, self.source_sequence))
        else:
            check_valid_sequence_of_sides(sounds)
            self.sound_sequence = sounds

    def __str__(self):
        return f"object of type {self.__class__} \n sources: {self.source_sequence} \n sounds: {self.sound_sequence} \n"

    def _generate_sound_from_source(self, source):
        """
        Generates a random sound location for a given source

        Args:
            source (str): an element from SIDES

        Returns:
            str: a side

        """
        check_valid_side(source)

        same_side = bernoulli.rvs(self.likelihood_same_side)

        return source if same_side else switch_side(source)

    def generate_source_sequence(self, init=None):
        """
        Generate a sequence of sources

        todo: might be computationally inefficient

        Args:
            init: initial source side (should be member of SIDES). If None, picked according to prior

        Returns:
            sequence of source sides

        """
        if init is None:
            sides, prior = [], []

            # for loop needed because SIDES is a set
            for s in SIDES:
                sides += [s]
                prior += [self.source_prior[s]]

            init = np.random.choice(sides, p=prior)

        check_valid_side(init)
        sequence = [init]
        generated_trials = 1

        while generated_trials < self.num_trials:
            change_point = bernoulli.rvs(self.hazard)
            last_source = sequence[-1]
            new_source = switch_side(last_source) if change_point else last_source
            sequence.append(new_source)
            generated_trials += 1

        return sequence


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

    sources_prior = {'left': .5, 'right': .5}
    """prior expectations about most likely side of a source"""

    def __init__(self, stimulus_object):
        """
        Args:
            stimulus_object (StimulusBlock): a stimulus with fixed hazard rate
        """
        self.stimulus_object = stimulus_object
        self.observations = None  # will be set by the self.observe method

    def observe(self, list_of_sounds=None):
        """
        Generate subjective observations of a given stimulus

        todo: not clear yet whether list_of_sounds should be automatically extracted from the stimulus_object attribute

        Args:
            list_of_sounds (list): list of sound locations (str). If None, uses self.stimulus_object.sound_sequence

        Returns:
            None: But sets self.observations

        """
        if list_of_sounds is None:
            list_of_sounds = self.stimulus_object.sound_sequence
        else:
            check_valid_sequence_of_sides(list_of_sounds)  # exception raised if a stimulus is invalid

        def apply_sensory_noise(side):
            """
            Intermediate function that switches sound location according to sensory noise

            Args:
                side (str): a member of SIDES

            Returns:
                perceived side after sensory noise is applied

            """
            return switch_side(side) if bernoulli.rvs(self.mislocalization_noise) else side

        self.observations = list(map(apply_sensory_noise, list_of_sounds))

        return None

    def process(self, observations=None, hazard=None):
        """
        This is where the bulk of the decision process occurs. Observations are converted into a decision variable.

        For now, only the log posterior odds of the sources is computed, and hazard rate is assumed fixed.

        Args:
            observations (list): sequence of perceived sound locations. If None, self.observations is used
            hazard: hazard rate, if None, the one from the stimulus_object attribute is fetched

        Returns:
            generator object that yields (log posterior odds, decisions)

        """
        if observations is None:
            observations = self.observations
        else:
            check_valid_sequence_of_sides(observations)  # exception raised if a stimulus is invalid

        if hazard is None:
            hazard = self.stimulus_object.hazard
        assert (isinstance(hazard, float) or isinstance(hazard, int)) and (0 <= hazard <= 1)

        prob_same_side = self.stimulus_object.likelihood_same_side

        # jump in accrued evidence towards 'right' choice if sound on right
        jump_magnitude = np.log(prob_same_side / (1 - prob_same_side))

        def discount_old_evidence(y):
            """
            Discount evidence from last time point in optimal sequential decision making in changing environment

            hazard rate is assumed known

            Args:
                y: evidence (log posterior odds) of previous time step

            Returns:
                float: positive favors 'right', negative 'left'

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

                yield log_posterior_odds, self._decide(log_posterior_odds)
                decision_number += 1

        return recursive_update()

    def _decide(self, decision_variable):
        """
        Makes a decision on a single trial, based on the decision variable

        Args:
            decision_variable: for now, log posterior odds

        Returns:
            str: an element from SIDES

        """
        s = np.sign(decision_variable)
        if s == -1:
            return 'left'
        elif s == 1:
            return 'right'
        elif s == 0:
            return 1 if bernoulli.rvs(self.bias) else 0


class Audio2AFCSimulation:
    """
    Use this class to launch simulations of our models
    """

    def __init__(self, tot_trials, h_values, meta_k, meta_prior_h):
        self.tot_trials = tot_trials

        assert isinstance(self.tot_trials, int) and self.tot_trials > 0

        # the following line implicitly checks that h_values is not a flot nor an int
        # i.e. raises TypeError.
        assert len(h_values) == len(meta_prior_h)

        self.h_values = h_values
        self.meta_k = meta_k
        self.meta_prior_h = meta_prior_h

        # todo: not sure yet how to handle blocks of stimuli and observer
        # self.stimulus
        # self.observer

        sources, sounds, hazards = [], [], []

        for block in self.generate_stimulus_blocks():
            sources += block.source_sequence
            sounds += block.sound_sequence
            hazards += [block.hazard] * block.num_trials

        self.data = pd.DataFrame({
            'source': sources,
            'source_switch': list(flag_change_points(sources)),
            'sound': sounds,
            'sound_switch': list(flag_change_points(sounds)),
            'hazard': hazards,
            'hazard_switch': list(flag_change_points(hazards)),
            })

    def generate_stimulus_blocks(self):
        """
        generate consecutive blocks of stimulus in which hazard rate is constant

        The hazard rate at the beginning of each block is sampled from self.meta_prior_h, excluding the hazard rate from
        the previous block. For the first block, no hazard rate value is excluded.
        The first source of each block is sampled by applying the new hazard rate to the last source from the previous
        block. For the very first block, the source is sampled from StimulusBlock.source_prior

        Returns:
            generator object that yields StimulusBlock objects
        """
        trials_generated = 0  # counter

        # initialize hazard rate and source for first trial of first block. None defaults to sampling from priors
        hazard = None
        first_source, last_source = None, None

        while trials_generated < self.tot_trials:

            # sample new hazard
            hazard = self.sample_meta_prior_h(hazard)

            # sample block length (sequence of trials with fixed hazard rate)
            block_length = get_next_change_point(self.meta_k)

            # reduce block_length if overshoot
            if block_length + trials_generated > self.tot_trials:
                block_length = self.tot_trials - trials_generated

            # pick first source of new block
            if last_source is not None:
                first_source = switch_side(last_source) if bernoulli.rvs(hazard) else last_source

            # generate the new block
            block = StimulusBlock(block_length, hazard, first_source=first_source)

            # what the generator yields
            yield block

            # update the counter
            trials_generated += block_length

            # update last source from last block (for next iteration of loop)
            last_source = block.source_sequence[-1]

    def sample_meta_prior_h(self, current_h=None):
        """
        Sample a new hazard rate value from the hyper-prior, excluding the current_h

        todo: check that the sampling statistics are the desired ones

        Args:
            current_h (int or float): if None, no value of h is excluded from the sampling set

        Returns:
            a sample from self.h_values, excluding current_h

        """
        if current_h is None:
            values, prior = self.h_values, self.meta_prior_h
        else:
            assert current_h in self.h_values
            values, prior = [], []
            normalization_constant = 0
            for j, h in enumerate(self.h_values):
                if h != current_h:
                    values.append(h)
                    p = self.meta_prior_h[j]
                    prior.append(p)
                    normalization_constant += p
            # normalize prior so that it adds up to 1
            prior = list(map(lambda x: x / normalization_constant, prior))

        return np.random.choice(values, p=prior)
