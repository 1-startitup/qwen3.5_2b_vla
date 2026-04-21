"""Test EMA semantics: after a few steps, current params != EMA shadow;
swap_in loads shadow into model; swap_out restores; state_dict round-trips.
"""
import sys; sys.path.insert(0, ".")
import torch
import torch.nn as nn
from train import EMA


def main():
    torch.manual_seed(0)
    m = nn.Linear(4, 4).cuda()
    ema = EMA(m, decay=0.9, offload_to_cpu=False)

    # Simulate 10 SGD steps that move params in one direction.
    with torch.no_grad():
        for _ in range(10):
            m.weight.data += 0.1
            m.bias.data += 0.1
            ema.update(m)

    # Shadow should lag behind: its value at t should be < current (because initial EMA=1.0 post-Linear-init, updates 0.1*step)
    current_w = m.weight.detach().clone()
    ema_w = ema.shadow["weight"]
    diff = (current_w - ema_w).abs().mean().item()
    print(f"current vs ema mean-abs-diff : {diff:.4f}   (expect > 0; EMA lags current)")
    assert diff > 1e-4, "EMA shadow did not diverge from current params"

    # Swap in EMA.
    backup = ema.swap_in(m)
    assert torch.allclose(m.weight, ema_w), "swap_in did not copy EMA into model"
    print("swap_in              : OK (model weights == EMA shadow)")

    # Swap out.
    ema.swap_out(m, backup)
    assert torch.allclose(m.weight, current_w), "swap_out did not restore current params"
    print("swap_out             : OK (model weights restored to pre-swap)")

    # State dict round-trip.
    st = ema.state_dict()
    ema2 = EMA(m, decay=0.9, offload_to_cpu=False)
    ema2.load_state_dict(st)
    for k in ema.shadow:
        assert torch.allclose(ema.shadow[k].cpu(), ema2.shadow[k].cpu()), f"round-trip failed for {k}"
    print("state_dict roundtrip : OK")

    print("\n>>> EMA semantics PASSED <<<")


if __name__ == "__main__":
    main()
