import torch
from torch.testing import assert_close
import pytest

from octopus.policy.loss import reinforce_loss

@pytest.mark.parametrize(
    "log_probs, rewards, discount_factor, expected_output",
    [
        (
            [torch.tensor(0.1).log(), torch.tensor(0.2).log(), torch.tensor(0.3).log()],
            [torch.tensor(1), torch.tensor(2), torch.tensor(3)],
            0.99,
            torch.tensor(8.4143)
        ),
        (
            torch.tensor([0.1, 0.2, 0.3]).log(),
            torch.tensor([1, 2, 3]),
            0.99,
            torch.tensor(8.4143)
        )
    ]

)
def test_calculate_reinforce_loss(log_probs, rewards, discount_factor, expected_output):
    output = reinforce_loss(log_probs, rewards, discount_factor)

    assert_close(output, expected_output, rtol=1e-5, atol=1e-5)