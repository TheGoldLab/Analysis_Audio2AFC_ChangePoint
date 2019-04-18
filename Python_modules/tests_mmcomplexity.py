"""
This is a test module to test the mmcomplexity module with unittest
Main reference: https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUp
"""
import unittest
import mmcomplexity as mmx


class TestModuleFunctions(unittest.TestCase):
    def test_allowed_sides(self):
        self.assertEqual(mmx.SIDES, {'left', 'right'})

    def test_check_valid_side(self):
        self.assertIsNone(mmx.check_valid_side('left'))
        self.assertIsNone(mmx.check_valid_side('right'))

        self.assertRaises(ValueError, mmx.check_valid_side, 0)
        self.assertRaises(ValueError, mmx.check_valid_side, 1)
        self.assertRaises(ValueError, mmx.check_valid_side, True)
        self.assertRaises(ValueError, mmx.check_valid_side, False)
        self.assertRaises(TypeError, mmx.check_valid_side, ['left'])


class TestAudioStimulus(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_number_of_trials(self):
        pass


class TestIdealObserverModel(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
