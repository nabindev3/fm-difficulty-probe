# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub's **"Report a vulnerability"**
button (Security tab → Private vulnerability reporting) on
https://github.com/nabindev3/fm-difficulty-probe, or by email to
nabin.dev33@gmail.com. Expect an acknowledgement within a week. Please do not
open a public issue for an exploitable problem.

## Supported versions

This is research code accompanying a paper. Only the latest commit on `main`
(and the most recent release tag) is supported; there are no backports.

## Threat model — what this code does and doesn't do

The pipeline is offline batch analysis: no network service, no untrusted user
input at runtime. The realistic attack surface is **files the pipeline loads**:

- **Model / SAE checkpoints (`*.pt`)** — loaded with
  `torch.load(..., weights_only=True)` everywhere, which refuses arbitrary
  pickle payloads. Do not weaken this if you fork the code.
- **Activation tensors (`*.safetensors`)** — the safetensors format cannot
  carry executable payloads by design.
- **Metadata (`*.parquet`, `*.yaml`)** — parsed with pyarrow and
  `yaml.safe_load` respectively; no arbitrary-object deserialization.
- **HuggingFace downloads** — `reproduce.sh` pulls Pythia-410m,
  Chronos-T5-small, and the artifact dataset from their canonical public
  repos; pin or mirror them if your environment requires provenance control.

Only run the pipeline on checkpoints and datasets you trust: even with the
mitigations above, loading attacker-controlled research artifacts is out of
scope of what this project can defend against.

## Dependency monitoring

- Dependabot watches the pip dependencies and GitHub Actions for known
  vulnerabilities and stale versions (`.github/dependabot.yml`).
- CI runs `pip-audit` against `requirements.lock` on every push (non-blocking:
  the lock is frozen for bit-reproducibility of the paper's numbers, so an
  advisory in a pinned version surfaces as a loud warning rather than a red X;
  fixes land as deliberate pin bumps).
