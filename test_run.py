import unittest
import torch
from run import AddModule, Head, N_EMBD, BLOCK_SIZE

class TestRun(unittest.TestCase):
    def test_nothing(self):
        self.assertTrue(True)

    def test_add_module(self):
        add_module = AddModule()
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        result = add_module(x, y)
        self.assertTrue(torch.equal(result, torch.tensor([5, 7, 9])))

    def test_head_module_causality(self):
        torch.manual_seed(1337)
        head_size = 16
        head = Head(head_size)
        head.eval() # Disable dropout for deterministic output

        B, T, C = 1, BLOCK_SIZE // 2, N_EMBD # Use a smaller block size for the test
        x = torch.randn(B, T, C)

        # Create a second input that's identical to the first, but with a
        # large change at a future timestep.
        future_timestep = T // 2
        x_modified = x.clone()
        x_modified[:, future_timestep:, :] += 10.0 # Add a large disturbance

        # Get the output for both original and modified inputs
        output_original = head(x)
        output_modified = head(x_modified)

        # 1. Test for Causality:
        # The output for all timesteps *before* the modification should be identical.
        # Due to floating point arithmetic, we use allclose instead of equal.
        self.assertTrue(
            torch.allclose(output_original[:, :future_timestep, :], output_modified[:, :future_timestep, :]),
            "Causality test failed: Output at past timesteps was affected by a future change."
        )

        # 2. Test for Influence:
        # The output for timesteps *at and after* the modification should be different.
        self.assertFalse(
            torch.allclose(output_original[:, future_timestep:, :], output_modified[:, future_timestep:, :]),
            "Influence test failed: Output at future timesteps was NOT affected by a future change."
        )

        # 3. Test output shape (as before)
        self.assertEqual(output_original.shape, (B, T, head_size))


if __name__ == '__main__':
    unittest.main()
