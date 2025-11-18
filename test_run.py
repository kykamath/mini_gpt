import unittest
import torch
from run import AddModule

class TestRun(unittest.TestCase):
    def test_nothing(self):
        self.assertTrue(True)

    def test_add_module(self):
        add_module = AddModule()
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        result = add_module(x, y)
        self.assertTrue(torch.equal(result, torch.tensor([5, 7, 9])))

if __name__ == '__main__':
    unittest.main()
