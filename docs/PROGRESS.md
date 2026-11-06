# Haladás

## 1. fázis (kódfelderítés): kész
- `docs/DESIGN.md` megírva és külön commitban (`2c2835129`).
- `docs/baseline.md` bővítve a NAS-összevetés szakasszal.
- Upstream keresés: nincs kapcsolódó PR; a #31538 tartalma nem olvasható (a `gh issue`/`gh api` tiltott).

## Következő: 2. fázis (megvalósítás)
Sorrend: `ggml_backend_sched_set_weight_alt` -> loader (ikertenzor) -> model/params/arg -> context bekötés.
