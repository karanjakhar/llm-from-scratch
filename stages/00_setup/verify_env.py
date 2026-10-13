"""Stage 00 — verify the environment and validate the VRAM mental model.

Two jobs:
  1. Confirm torch/CUDA are working and report the hardware.
  2. Test the claim from docs/00_setup.md that training an N-parameter model
     costs ~16 bytes per parameter in fixed memory (fp32 params + fp32 grads
     + two fp32 AdamW buffers), independent of batch size.

If the prediction and the measurement disagree by much, the mental model is
wrong and that is worth understanding before Stage 01.

Run:  uv run python stages/00_setup/verify_env.py
"""

import torch
import torch.nn as nn

MB = 1024 ** 2


def report_hardware() -> torch.device:
    print("=" * 62)
    print("ENVIRONMENT")
    print("=" * 62)
    print(f"  torch            {torch.__version__}")
    print(f"  cuda available   {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n  No CUDA. Everything in this repo runs on CPU but slowly.")
        print("  Stages 00-07 are fine on CPU. Stage 08 onward wants a GPU.")
        return torch.device("cpu")

    props = torch.cuda.get_device_properties(0)
    print(f"  cuda version     {torch.version.cuda}")
    print(f"  device           {props.name}")
    print(f"  total VRAM       {props.total_memory / MB:,.0f} MB")
    print(f"  compute capab.   {props.major}.{props.minor}")
    print(f"  bf16 supported   {torch.cuda.is_bf16_supported()}")
    print(f"  SMs              {props.multi_processor_count}")
    return torch.device("cuda")


class ToyModel(nn.Module):
    """A plain MLP stack. Not a transformer — the point is a known parameter
    count with nothing else going on, so the memory arithmetic is unambiguous."""

    def __init__(self, width: int = 1024, depth: int = 12):
        super().__init__()
        # bias=False keeps the parameter count exactly depth * width^2
        self.layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(depth)
        )

    def forward(self, x):                      # x: (B, width)
        for layer in self.layers:
            x = torch.relu(layer(x))           # (B, width)
        return x


def measure_training_memory(device: torch.device, width=1024, depth=12, batch=8):
    print()
    print("=" * 62)
    print("VRAM MODEL CHECK")
    print("=" * 62)

    if device.type != "cuda":
        print("  Skipped: needs CUDA to measure allocator stats.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    model = ToyModel(width, depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    after_params = torch.cuda.memory_allocated()

    # fp32 everywhere here so the arithmetic is simple and checkable:
    #   params 4 + grads 4 + AdamW exp_avg 4 + exp_avg_sq 4 = 16 bytes/param
    predicted = n_params * 16

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(batch, width, device=device)
    loss = model(x).square().mean()
    loss.backward()
    after_backward = torch.cuda.memory_allocated()

    # AdamW allocates its two state buffers lazily, on the first step.
    opt.step()
    after_step = torch.cuda.memory_allocated()

    params_mb = (after_params - baseline) / MB
    grads_mb = (after_backward - after_params) / MB   # includes some activations
    state_mb = (after_step - after_backward) / MB
    total_mb = (after_step - baseline) / MB
    peak_mb = torch.cuda.max_memory_allocated() / MB

    print(f"  model            width={width}, depth={depth}, batch={batch}")
    print(f"  parameters       {n_params:,} ({n_params / 1e6:.2f}M)")
    print()
    print(f"  predicted fixed  {predicted / MB:,.1f} MB   (16 bytes/param)")
    print(f"  measured total   {total_mb:,.1f} MB")
    print(f"  peak allocated   {peak_mb:,.1f} MB   (incl. activations)")
    print()
    print("  breakdown as measured by the allocator:")
    print(f"    params         {params_mb:,.1f} MB   (expect 4 bytes/param)")
    print(f"    grads + acts   {grads_mb:,.1f} MB   (grads are 4 bytes/param)")
    print(f"    AdamW state    {state_mb:,.1f} MB   (expect 8 bytes/param)")

    err = abs(total_mb - predicted / MB) / (predicted / MB) * 100
    print()
    print(f"  prediction error {err:.1f}%")
    if err < 20:
        print("  → mental model holds. The 16 bytes/param rule is usable.")
    else:
        print("  → mismatch >20%. Find out why before Stage 01; the extra is")
        print("    probably activations held for backward. Re-read docs/00_setup.md §2.")

    # The ablation from docs/00_setup.md: a budget that forgets AdamW states
    # is wrong by 50%, which is the difference between 'fits' and 'OOM'.
    print()
    print(f"  forgot AdamW?    {n_params * 8 / MB:,.1f} MB predicted vs "
          f"{total_mb:,.1f} MB real → off by 2x")

    del model, opt, x, loss
    torch.cuda.empty_cache()


def budget_table(total_vram_mb: float):
    print()
    print("=" * 62)
    print("WHAT FITS ON THIS GPU (full fine-tune, 16 bytes/param)")
    print("=" * 62)
    print(f"  {'params':>10}  {'fixed cost':>12}  {'% of VRAM':>10}  verdict")
    for n in (10e6, 50e6, 125e6, 350e6, 1e9, 7e9):
        mb = n * 16 / MB
        pct = mb / total_vram_mb * 100
        if pct < 30:
            verdict = "comfortable"
        elif pct < 70:
            verdict = "workable"
        elif pct < 95:
            verdict = "tight — needs checkpointing"
        else:
            verdict = "no — LoRA or inference only"
        print(f"  {n / 1e6:>9.0f}M  {mb:>9,.0f} MB  {pct:>9.1f}%  {verdict}")
    print()
    print("  Note: leave ~600 MB for the CUDA context and fragmentation, and")
    print("  remember activations are on top of all of this.")


def main():
    device = report_hardware()
    measure_training_memory(device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / MB
        budget_table(total)
    print()
    print("Stage 00 done. Next: docs/01_tokenizer.md")


if __name__ == "__main__":
    main()
