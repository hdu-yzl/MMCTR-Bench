import torch
import torch.nn.functional as F

from mmctr.analysis.legacy_alignment import LegacyAlignmentObjective


def test_kl_log_domain_matches_value_and_both_gradients():
    torch.manual_seed(4)
    anchor = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    other = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    objective = LegacyAlignmentObjective("kl", ["id", "text"], 8, 0.3)
    actual = objective({"id": anchor, "text": other})
    expected = 0.3 * F.kl_div(
        F.log_softmax(other, -1), F.softmax(anchor, -1), reduction="batchmean"
    )
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual, (anchor, other), retain_graph=True)
    references = torch.autograd.grad(expected, (anchor, other))
    for actual_grad, expected_grad in zip(gradients, references):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_kl_underflow_has_finite_value_and_gradients():
    anchor = torch.tensor([[100.0, -100.0, 0.0]], requires_grad=True)
    other = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    loss = LegacyAlignmentObjective("kl", ["id", "text"], 3, 0.3)({"id": anchor, "text": other})
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(anchor.grad).all()
    assert torch.isfinite(other.grad).all()


def test_wide_batchnorm_matches_normal_values_and_gradients():
    from mmctr.analysis.numerical_stability import WideBatchNorm1d

    torch.manual_seed(9)
    original = torch.nn.BatchNorm1d(8)
    stable = WideBatchNorm1d(original)
    x = torch.randn(16, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = original(x), stable(y)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    weights = torch.randn_like(a)
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(original.weight.grad, stable.weight.grad, rtol=1e-5, atol=1e-6)
    original.eval()
    stable.eval()
    torch.testing.assert_close(original(x), stable(y), rtol=1e-5, atol=1e-6)


def test_wide_batchnorm_extreme_finite_inputs_and_running_variance():
    from mmctr.analysis.numerical_stability import WideBatchNorm1d

    stable = WideBatchNorm1d(torch.nn.BatchNorm1d(4))
    values = (torch.arange(32, dtype=torch.float32).reshape(8, 4) * 1e22).requires_grad_()
    output = stable(values)
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(values.grad).all()
    assert torch.isfinite(stable.running_var).all()
    assert stable.running_var.max() > torch.finfo(torch.float32).max
    assert stable.weight.dtype == torch.float32 and output.dtype == torch.float32
    stable.eval()
    assert torch.isfinite(stable(values)).all()
