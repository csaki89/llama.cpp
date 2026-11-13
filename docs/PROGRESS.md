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
- 2. rész (dinamikus KV): kész. `n_ctx_init` cellával indul, darabokban (64 MiB) nő, kiürülve csökken; statikus elhelyezéssel a kimenet bájtra azonos (`0063c4c5`), a KV a VRAM-ban 416 helyett 32 MiB (2560 cella). A VMM-út nem készült el (DECISIONS.md 22).
- 3. rész (zónaváltási szabály): kész. A KV növekedésével a határhoz legközelebbi rétegek lekerülnek, kiürüléskor (új beszélgetés) a KV csökken és a rétegek visszakerülnek; állapotbetöltés után is könyvel. Ahol a kérés alatt a hely nem változik, a kimenet bájtra azonos a statikus elhelyezéssel (`-ngl 11 --dual-buffer`: `f773f5bc`). DECISIONS.md 24-25.
- Ellenőrzés és dokumentáció: test-results.md "Automatikus háromzónás működés"; SUMMARY.md 7. (Ollama-patch, NAS-teendők); DESIGN-3zone.md 6.5. Kikapcsolva a baseline változatlan. A 12,8k tokenes sebességsor nem futott le (a rendszer kevés memória miatt leállította a mérést).

## Háromzónás terv, valódi használat (MoE, VMM, FA, Ollama)
- 1. MoE (részleges flex rétegek a fit override-jaiból): kész, `granite-3.1-1b-a400m`-mel kipróbálva (DECISIONS.md 26).
- 2. VMM a KV-növeléshez (ggml-cuda/HIP növelhető puffer, rétegenként, tartalék úttal, csúcs a zónaszabályban): kész, a laptopon szimulációval ellenőrizve, a ROCm-build az ollama-dualbuf CI-jén (DECISIONS.md 27-29).
- 3. Flash Attention: forrásból kiderítve, nem kényszerítjük, egyértelmű napló (DECISIONS.md 30).
- 8. VMM-laphiba: a leképezés aszinkron (DRM `GEM_VA`), KFD-korlát + várakozás, `commit` lépés, `vmm-stress` NAS-teszt (DECISIONS.md 33-36).
- A ROCm-build nem futott le: a csaki89/ollama repóban ki vannak kapcsolva az Actions (DESIGN-3zone.md 7.2).
- 9. A VMM-teszt a valódi megvalósítást is hívja (`vmm-stress-impl`), a tesztprogramok a ROCm-image részei (DECISIONS.md 37). Az image elkészült: `0.40.2-dualbuf-b1798d9` (DESIGN-3zone.md 7.2), a NAS-os futtatás hátravan.
- 10. VRAM-visszaadás és gyors VMM-teszt (DECISIONS.md 38-39): élő aréna-/leképezés-számláló a ggml-cudában, a teszt a VRAM visszatérését időzítve méri; az első hozzáférés csak az új darabokat éri, mért késleltetéssel.
- 11. HIP: csak a KFD-korlát (a vak várakozás csak hibakereső mód), a korlát felszabadítása a commit szinkronja után, a commit hibájánál másolásos tartalék (DECISIONS.md 40); a TrueNAS 27 / 6.18-as kernel vizsgálata a ROADMAP-ban.
