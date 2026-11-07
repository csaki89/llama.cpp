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

## 5. fázis (zárás): kész
- `docs/SUMMARY.md` megírva. Az éjszakai munka vége; a branch nincs pusholva.

## Háromzónás terv, rugalmas sáv (`--flex-layers`)
- 1. építőkocka: rétegek áthelyezése futás közben, K/V követése: kész (`3e60401dc`, `a6dd34d00`).
- Befejezés, 1. lépés (a sáv helye): kész. A sáv a GPU-s rétegek első N rétege, a levétel a határ felől halad; a `dev_layer()` levett rétegnél a CPU (DECISIONS.md 15-16). Levett rétegekkel a generálási gráf 2 hasítású (0, 1, 2 levett réteg: 2 / 2 / 2).
- 2. lépés (a 2. zóna host-példánya): kész. Levett rétegnél op offload a host-példányból, a visszatétel a host-példányból (4-7 ms/réteg, fájlból 23-106 ms). Egy végig levett réteg kimenete bájtra azonos a `-ngl 13 --dual-buffer` futáséval (DECISIONS.md 17-18).
- 3. lépés (másolási sebesség): kész. K/V GPU -> CPU 32K kontextusnál 149-163 ms -> 12-15 ms (a másolás 140-153 ms -> 0,5 ms: 1628 használt cella, DMA rögzített memóriába; a maradék a foglalás és a törlés). Részletek: test-results.md (4. lépés), DECISIONS.md 19-20.
- 4. lépés (ellenőrzés, dokumentáció): kész. test-results.md "A rugalmas sáv befejezése"; DESIGN-3zone.md (2.1, 2.3, 2.6, 2.7, kockázatok, Következő lépések), ROADMAP.md, SUMMARY.md (6. NAS-on). Opció nélkül a baseline változatlan.

## Háromzónás terv, automatikus működés (`--auto-zones`)
- Terv: DESIGN-3zone.md 6. (`ba9a60e1b`).
- 1. rész (a sáv mérete automatikusan): kész. Két fit (max és kis kontextus), `flex_ngl_full` modellparaméter, `n_ctx_init` / `zone_vram_margin` kontextparaméter (a 2-3. rész használja). DECISIONS.md 21.
