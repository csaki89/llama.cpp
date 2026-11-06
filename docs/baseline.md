# Alapmérés a laptopon (eredeti kód, kódmódosítás nélkül)

Cél: a NAS-on mért jelenség (CLAUDE.md, "Mért tények") mechanizmusának ellenőrzése a laptopon. Az iGPU ugyanazt a RAM-ot használja, mint a CPU, ezért a sebességarányok nem vihetők át a NAS-ra. A pufferek és a mechanizmus számítanak.

## Beállítás
- Gép: i5-1145G7 (4 szál használva a 8-ból, alapértelmezett), Intel Iris Xe (Vulkan0), 15,7 GB RAM, Windows 11, natív MSVC build.
- Kód: a ggml-org/llama.cpp `master` `e14a77971` állapota, módosítás nélkül (a branch csak docs/tools/CLAUDE.md fájlokat tesz hozzá).
- Modell: Qwen2.5-1.5B-Instruct Q4_K_M, 29 réteg (28 + kimeneti). Részleges offload: `-ngl 14`.
- Közös kapcsolók: `--load-mode none --log-verbosity 4 -fa auto -b 512 -ub 512 --temp 0 --seed 1 -n 200 --ignore-eos -no-cnv`
- Promptok: rövid (110 token) és hosszú (1611 token), nyers szöveg, a szkript generálja. Minden mód/prompt 3 ismétléssel, egy eldobott bemelegítő futás után.
- Környezet: a `LLAMA_*` változók száma 0 volt (a szkript ellenőrzi és törli őket).
- Eszköz: `tools/dualbuf/bench.sh` (módok: `tools/dualbuf/modes.conf`), összesítés: `tools/dualbuf/summarize.sh`. Új mód (pl. dual-buffer) egy új sor a `modes.conf`-ban. A szkript bash, Linuxon is fut.
- A generálás 199 mért tokent jelent (a 200. token a prompt feldolgozásából jön).

## Eredmények
Átlag (min-max) 3 ismétlésre, tok/s.

| mód | prompt | prompt tok | prompt t/s | gen t/s | kimenet md5 | betöltött pufferek (MiB) |
|---|---|---|---|---|---|---|
| repack (`--no-host`) | rövid | 110 | 139,5 (130,9-143,8) | 20,8 (17,4-22,8) | 8e2d6759 | CPU 225,15; CPU_REPACK 308,18; Vulkan0 526,55 |
| repack (`--no-host`) | hosszú | 1611 | 130,1 (128,8-132,2) | 18,3 (15,7-19,6) | 2be92339 | ua. |
| host-puffer | rövid | 110 | 290,2 (285,3-295,2) | 20,8 (17,9-22,4) | 8e2d6759 | Vulkan0 526,55; Vulkan_Host 533,33 |
| host-puffer | hosszú | 1611 | 321,8 (309,4-328,8) | 19,2 (18,8-19,5) | 0063c4c5 | ua. |
| host + `--no-op-offload` | rövid | 110 | 119,4 (105,6-133,3) | 22,1 (22,0-22,2) | 8e2d6759 | Vulkan0 526,55; Vulkan_Host 533,33 |
| host + `--no-op-offload` | hosszú | 1611 | 130,5 (107,1-144,8) | 18,4 (17,8-19,6) | df42ead7 | ua. |
| repack + `--no-op-offload` | rövid | 110 | 147,2 (131,2-155,9) | 22,3 (21,8-22,7) | 8e2d6759 | CPU 225,15; CPU_REPACK 308,18; Vulkan0 526,55 |
| repack + `--no-op-offload` | hosszú | 1611 | 146,4 (140,7-150,9) | 18,8 (17,4-19,9) | 2be92339 | ua. |
| csak CPU (`-dev none -ngl 0`) | rövid | 110 | 97,2 (88,2-104,6) | 25,3 (25,0-25,6) | 8e2d6759 | CPU 463,35; CPU_REPACK 596,53 |
| csak CPU | hosszú | 1611 | 83,9 (82,6-85,0) | 20,5 (20,3-20,9) | f773f5bc | ua. |
| teljes GPU (`-ngl 99`) | rövid | 110 | 344,3 (337,7-355,9) | 21,0 (20,9-21,1) | 8e2d6759 | Vulkan0 934,69; Vulkan_Host 125,19 |
| teljes GPU | hosszú | 1611 | 402,6 (400,5-406,6) | 20,2 (20,1-20,3) | a8939715 | ua. |

A kimenet első ~200 karaktere: a rövid promptnál minden módban ugyanaz (" The history of computing is a story of steady abstraction. Early machines were programmed ..."). A hosszú promptnál a `repack`, `host`, `host_noopoff` és `gpu_full` első 200 karaktere azonos (" The text is a list of sections that all contain the same sentence: ..."), a `cpu_only` már a 32. bájtnál másképp indul (" The text is a list of sections, each containing the same sentence with slight variations. ..."). Az ismétlések egy módon belül bájtra azonosak.

## Mit látunk
1. **A pufferek megjelennek, ahogy a CLAUDE.md írja.** `--no-host` nélkül a CPU-n maradó súlyok a `Vulkan_Host` pufferbe kerülnek (533,33 MiB), `CPU_REPACK` nincs. `--no-host`-tal ugyanez a rész `CPU_REPACK` (308,18) + `CPU` (225,15) = 533,33 MiB, `Vulkan_Host` nincs. A `CPU` puffer a nem repackelt tenzorokat tartalmazza (benne a ~125 MiB-os token embedding, lásd 6.).
2. **A host-puffer promptja op offloaddal ~2,4-szer gyorsabb, mint nélküle:** 290,2 vs 119,4 (rövid), 321,8 vs 130,5 (hosszú). Ez egyezik a NAS-os következtetéssel (az előny az op offloadból jön).
3. **Repack esetén az op offload nem számít:** `repack` 139,5 / 130,1 vs `repack_noopoff` 147,2 / 146,4. A különbség a zajon belül van (a hosszúnál a `repack` kicsit lassabb is, ezt nem értjük). Ez egyezik azzal, hogy a CPU_REPACK tenzorokat az op offload nem tudja a GPU-ra küldeni. A host-puffer promptja 2,1-2,5-szerese a repackének, a NAS-on ez +23...+64% volt.
4. **A generálási előny NEM reprodukálódik.** A NAS-on a repack generálása sokkal gyorsabb volt a host-pufferénél (pl. 30 rétegnél 28,7-29,5 vs 20,2). Itt a repack és a host gen sebessége megkülönböztethetetlen (20,8 vs 20,8 rövidnél, 18,3 vs 19,2 hosszúnál, a szórás 2-5 tok/s). Ez nem hiba, de a laptopon a dual-buffer generálási nyeresége nem mérhető. Feltevés (nem ellenőriztem): a generálás itt a megosztott RAM sávszélességén szűk keresztmetszet, és az iGPU is ezt használja. Erre utal, hogy a csak-CPU mód a leggyorsabb generálásban (25,3 / 20,5), a teljes GPU pedig nem gyorsabb nála (21,0 / 20,2).
5. **A kimenet nem egyezik minden módban.** Rövid prompt: mind a 6 mód bájtra azonos kimenetet ad. Hosszú prompt: a `repack` és a `repack_noopoff` azonos (ahogy várható, mert az op offload nem érinti), de a `host`, `host_noopoff`, `cpu_only` és `gpu_full` mind más. A `repack` és a `host`/`host_noopoff`/`gpu_full` a 365. bájtnál válik szét, a `cpu_only` már a 32.-nél. A `host` és a `host_noopoff` is eltér, vagyis az, hogy a mátrixszorzás a GPU-n vagy a CPU-n fut, számszerűen más eredményt ad. Ez lebegőpontos eltérésnek tűnik (temperature 0 mellett egy közel egyenlő logit más tokent választ), nem hibának. A dual-buffer helyességi tesztjénél ezért a "bájtra azonos" feltétel csak rövid prompton teljesül, vagy ha a két futás ugyanazon az eszközön számol. A hosszú promptnál az elvárt eltérést előre kell kezelni, például az első N token egyezésével.
6. **Teljes GPU mellett is marad `Vulkan_Host` (125,19 MiB).** Ez a token embedding (151936 x 1536 Q4_K, ~125,2 MiB: a méret egyezik), Hogy miért marad host pufferben, nem vizsgáltam.

## Korlátok
- 3 ismétlés, nagy szórás (különösen a gen sebességnél és a `host_noopoff` promptjánál), ezért a kisebb különbségek (pl. `repack` vs `repack_noopoff`) nem értelmezhetők.
- Az iGPU és a CPU közös RAM-ot és közös hőkeretet használ, a mérés sorrendje (repack, host, ...) befolyásolhatja az eredményt.
- A szkript első verziójában egy printf-hiba összeragasztotta a prompt nevét és a tokenszámot a TSV-ben. A szkriptet javítottam, a már lefutott mérés TSV-jét mechanikusan helyreállítottam (a nyers `.err`/`.out` fájlok érintetlenek, a számok nem változtak).

## NAS-összevetés (ugyanaz a modell: Qwen2.5-1.5B Q4_K_M, ROCm, RX 6700)
A NAS-os számokat a felhasználó adta meg, nem általam mért adatok. Prompt / generálás, tok/s, rövid / hosszú prompt.

| | NAS (ROCm, RX 6700) | Laptop (Vulkan, Iris Xe) | NAS / laptop |
|---|---|---|---|
| csak CPU, prompt | 466 / 482 | 97,2 / 83,9 | 4,8x / 5,7x |
| csak CPU, generálás | 39,2 / 37,5 | 25,3 / 20,5 | 1,5x / 1,8x |
| teljes GPU, prompt | 2230 / 3510 | 344,3 / 402,6 | 6,5x / 8,7x |
| teljes GPU, generálás | 160,6 / 155,4 | 21,0 / 20,2 | 7,6x / 7,7x |
| GPU / CPU, prompt | 4,8x / 7,3x | 3,5x / 4,8x | |
| GPU / CPU, generálás | 4,1x / 4,1x | 0,83x / 0,99x | |

Következmény: a NAS-on a GPU generálásban kb. 4-szer gyorsabb a CPU-nál, a laptopon nem gyorsabb (sőt lassabb). A laptopon ezért a dual-buffer generálási hatása nem mérhető, csak a helyesség (puffer, kimenet) és a prompt-gyorsulás. A generálási nyereséget a NAS-on kell mérni.
