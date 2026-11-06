# Haladás

## 1. fázis (kódfelderítés): kész
- `docs/DESIGN.md` külön commitban. `docs/baseline.md` bővítve a NAS-összevetéssel. Upstream keresés: nincs kapcsolódó PR.

## 2. fázis (megvalósítás): kész
- `--dual-buffer` / `LLAMA_ARG_DUAL_BUFFER`, alapból ki. Érintett fájlok: `ggml-backend.h/.cpp` (`ggml_backend_sched_set_weight_alt`), `llama-model-loader.h/.cpp`, `llama-model.h/.cpp`, `llama-context.cpp`, `llama.h`, `common/arg.cpp`, `common/common.h/.cpp`.
- Füstpróba (Qwen2.5-1.5B, `-ngl 14`): `dual-buffer: 87 weights also stored in Vulkan_Host (308.18 MiB)`, hosszú prompt ~326 tok/s.
- Megjegyzés: a settings.json `Bash(cmd //c tools/dualbuf/build-vulkan.bat)` allow-szabálya nem használható, mert a `cmd` perjeles útvonalat nem érti. Működő forma: `cmd //c "tools\dualbuf\build-vulkan.bat"` (ezt az allow-listán javítani kell).

## 3. fázis (ellenőrzés): kész
- Minden elfogadási feltétel teljesül, részletek: `docs/test-results.md` (commit `e2bf5e847` kódon).

## 4. fázis (korlátozás): kész
- `--dual-buffer-pattern`, `--dual-buffer-limit` (docs/test-results.md, 4. fázis szakasz). Alapviselkedés újraellenőrizve.

## Következő: 5. fázis (docs/SUMMARY.md)
