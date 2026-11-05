# Integration into LIFT-SuperMimic

Source: `https://github.com/DucksAreRoasted/SuperMimic_Reproduce.git`

Imported revision: `555d1b18232ef110de4526e221d2fb441068dd07`.

The import uses committed source at that revision; uncommitted deployment changes
from the original local working tree were not imported or modified. Media demos,
training logs, datasets and checkpoints were not copied.

The baseline is the `supermimic` member of the root uv workspace. Dependencies and
optional training/retargeting variants are maintained in the root `pyproject.toml`
and root `uv.lock`; `pyproject.upstream.toml` preserves the original package recipe.

Integration changes are restricted to:

- Packaging and root uv commands.
- Locating shared third-party code at root `third_party/`.
- Default datasets under root `data/`.
- Portable mesh/config paths and data-preparation/helper scripts.
- Test data paths and the README integration notice.

The baseline actor, critic, world model and training-update implementation are
preserved. Run baseline and TADG training from the workspace root using the same
`uv run --extra train --locked` environment. Formal training is reserved for the user's server; the user explicitly authorized
a local R1/LAFAN smoke run on 2026-11-05.

The original MIT license is retained as `LICENSE`. Other third-party assets retain
their original notices at the shared `third_party/` location.
