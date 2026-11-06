# Ellenőrzés: dual-buffer (3. fázis)

Teszt: `tools/dualbuf/bench.sh` (3 ismétlés, átlag és min-max), Qwen2.5-1.5B Q4_K_M, `-ngl 14`, `--load-mode none`, `LLAMA_*` környezeti változó nincs. Kód: `e2bf5e847` (dual-buffer) a `master` `e14a77971`-re építve. Újabb módok: `dual` (`--dual-buffer`) és `dual_noopoff` (`--dual-buffer --no-op-offload`).

## Eredmény (tok/s, átlag (min-max))
| mode | prompt | n | prompt tok | prompt t/s mean (min-max) | gen t/s mean (min-max) | output md5 | buffers (MiB)
|---|---|---|---|---|---|---|---
| repack | short | 3 | 110 | 136.8 (133.5-141.4) | 22.1 (22.0-22.1) | 8e2d6759 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55
| repack | long | 3 | 1611 | 129.5 (127.3-130.6) | 19.2 (18.8-19.5) | 2be92339 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55
| host | short | 3 | 110 | 268.2 (251.2-280.2) | 20.9 (20.8-21.0) | 8e2d6759 | Vulkan0=526.55 Vulkan_Host=533.33
| host | long | 3 | 1611 | 307.0 (303.9-309.2) | 18.2 (18.0-18.4) | 0063c4c5 | Vulkan0=526.55 Vulkan_Host=533.33
| host_noopoff | short | 3 | 110 | 117.0 (109.8-125.0) | 21.0 (20.6-21.4) | 8e2d6759 | Vulkan0=526.55 Vulkan_Host=533.33
| host_noopoff | long | 3 | 1611 | 131.1 (124.8-134.3) | 17.8 (16.5-18.6) | df42ead7 | Vulkan0=526.55 Vulkan_Host=533.33
| repack_noopoff | short | 3 | 110 | 129.5 (126.5-135.2) | 21.0 (20.6-21.3) | 8e2d6759 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55
| repack_noopoff | long | 3 | 1611 | 139.7 (138.9-140.2) | 18.4 (18.3-18.5) | 2be92339 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55
| cpu_only | short | 3 | 110 | 84.2 (82.1-86.3) | 23.3 (22.8-23.7) | 8e2d6759 | CPU=463.35 CPU_REPACK=596.53
| cpu_only | long | 3 | 1611 | 81.6 (80.4-82.6) | 19.1 (18.8-19.5) | f773f5bc | CPU=463.35 CPU_REPACK=596.53
| gpu_full | short | 3 | 110 | 334.4 (317.1-344.2) | 19.9 (19.7-20.1) | 8e2d6759 | Vulkan0=934.69 Vulkan_Host=125.19
| gpu_full | long | 3 | 1611 | 369.3 (366.1-374.8) | 18.9 (18.8-19.1) | a8939715 | Vulkan0=934.69 Vulkan_Host=125.19
| dual | short | 3 | 110 | 255.7 (249.2-259.1) | 20.9 (20.6-21.3) | 8e2d6759 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55 Vulkan_Host=308.18
| dual | long | 3 | 1611 | 304.5 (295.3-311.2) | 18.3 (17.8-18.6) | 35969901 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55 Vulkan_Host=308.18
| dual_noopoff | short | 3 | 110 | 127.7 (121.4-137.4) | 21.0 (20.4-21.3) | 8e2d6759 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55 Vulkan_Host=308.18
| dual_noopoff | long | 3 | 1611 | 140.2 (138.3-142.9) | 18.6 (18.4-18.8) | 2be92339 | CPU=225.15 CPU_REPACK=308.18 Vulkan0=526.55 Vulkan_Host=308.18

A futások közötti eltérés jelentős (a gép állapota óráról órára változik): az első alapmérés (`docs/baseline.md`) és ez a futás ugyanazon módoknál eltér (pl. `host` rövid prompt: 290 vs 268 tok/s). Ezért a sebességfeltételeket az **azonos futáson belüli** `host` és `repack` módokhoz viszonyítom.

## Elfogadási feltételek

| # | Feltétel | Eredmény |
|---|---|---|
| 1 | Kikapcsolva minden meglévő mód kimenete (md5) és puffere egyezik a baseline-nal | **Teljesül.** Mind a 6 régi mód mindkét prompton ugyanazt az md5-öt adja, mint a baseline-ban (rövid: `8e2d6759`; hosszú: repack `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, repack_noopoff `2be92339`, cpu_only `f773f5bc`, gpu_full `a8939715`), és a pufferek is azonosak (pl. `CPU 225,15; CPU_REPACK 308,18; Vulkan0 526,55`). Kikapcsolva a naplóban nincs `dual-buffer` sor. |
| 2 | `dual_noopoff`, hosszú prompt: kimenet = repack + no-op-offload (`2be92339`) | **Teljesül.** md5 `2be92339`, és a kimenet bájtra azonos a `repack_noopoff`-éval, mind a 3 ismétlésben. A CPU tehát a repackelt példánnyal számol. |
| 3 | `dual` op offloaddal: a prompt a host-puffer szintjén van, nem a repackén | **Teljesül.** Rövid: `dual` 255,7 (249,2-259,1) vs `host` 268,2 vs `repack` 136,8 → 95% a hosténak, 1,87x a repacké. Hosszú: `dual` 304,5 (295,3-311,2) vs `host` 307,0 vs `repack` 129,5 → 99% a hosténak, 2,35x a repacké. A min-max tartományok a hosszúnál átfednek a `host`-tal, a rövidnél kicsit alatta vannak. A `dual_noopoff` (127,7 / 140,2) a repack szintjén van, vagyis a gyorsulás tényleg az op offloadból jön, és az a host-példányt használja. |
| 4 | Rövid prompton minden mód kimenete bájtra azonos | **Teljesül.** mind a 8 mód md5 `8e2d6759`. |
| 5 | Mindkét puffer megjelenik a naplóban a várt mérettel; RAM-többlet mérve | **Teljesül.** `load_tensors: dual-buffer: 87 weights also stored in Vulkan_Host (308.18 MiB)`, `CPU_REPACK 308,18 MiB` és `Vulkan_Host 308,18 MiB`. A saját memóriabontás `Host` sora: repack 225 MiB, dual 533 MiB: a többlet **308 MiB**, a várt ~308 MiB-tal egyezik. (A `host` módban 533 MiB, vagyis a dual ugyanannyi host-memóriát használ, mint a host-puffer, a repack-példány pedig +308 MiB a `CPU_REPACK`-ben.) |

## Egyéb megfigyelések
- **Generálás:** `dual` 20,9 / 18,3, `repack` 22,1 / 19,2, `host` 20,9 / 18,2 tok/s. Nincs mérhető különbség a módok között (ahogy a baseline-ban sem), a laptopon a generálási hatás nem mérhető, a NAS-on kell mérni.
- **Hosszú prompt kimenete `dual` módban** (`35969901`) különbözik a többi módtól, de ismétlésenként determinisztikus. Ez várt, mert a prompt a GPU-n a host-példánnyal, a generálás a CPU-n a repackelttel számol. Az első eltérés a `repack`-hez képest a 365. bájtnál, a `host`/`gpu_full`-hoz képest a 421.-nél van, vagyis a kimenet eleje közös. Ugyanez a jelenség (lebegőpontos eltérés) van a meglévő módoknál is (`docs/baseline.md`).
- **Élhelyzetek** (24 token, rövid prompt, minden esetben ugyanaz a kimenet `6417fa59` és rc=0):
  - `-ngl 0 --dual-buffer`: 168 súly duplikálva (596,53 MiB), a kimenet azonos a `-ngl 0 --no-host`-éval.
  - `-ngl 99 --dual-buffer`: 0 súly duplikálva (nincs CPU-n maradó repackelhető súly), semmi nem változik.
  - `--load-mode mmap --dual-buffer`: figyelmeztetéssel kikapcsol ("requires --load-mode none"). Megjegyzés: az alapértelmezett `auto` mód ezen a gépen nem mmap, ott aktív.
  - `--no-repack --dual-buffer`: figyelmeztetéssel kikapcsol.
  - `-dev none --dual-buffer`: nincs host-puffer, figyelmeztetéssel kikapcsol.
  - `--dual-buffer --no-host`: a dual-buffer nyer (DECISIONS.md 3.).
  - `LLAMA_ARG_DUAL_BUFFER=1`: bekapcsolja.
- **Nem tesztelt:** MoE modell (`MUL_MAT_ID`), többeszközös gép, ROCm/CUDA, `llama-server` hosszú futás, LoRA. Ezeket a NAS-on / külön kell ellenőrizni.

## 4. fázis: a duplikálás korlátozása
Új, alapból kikapcsolt kapcsolók `--dual-buffer` mellett: `--dual-buffer-pattern REGEX` (a tenzornévre `regex_search`) és `--dual-buffer-limit MiB` (a másolatok összmérete; a tenzorok létrehozási, azaz réteg-sorrendjében vág, a korai rétegek kapják meg). Env: `LLAMA_ARG_DUAL_BUFFER_PATTERN`, `LLAMA_ARG_DUAL_BUFFER_LIMIT`. Minden szám **egyetlen futás** (nem 3 ismétlés), hosszú prompt (1611 token), `-ngl 14`, tájékoztató jellegű:

| beállítás | duplikált súly | Vulkan_Host (MiB) | prompt tok/s | kimenet md5 |
|---|---|---|---|---|
| nincs korlát (alapértelmezés) | 87 | 308,18 | 341,9 | 35969901 |
| `ffn_` | 36 | 265,78 | 227,2 | c2abe0a6 |
| `attn_` | 51 | 42,40 | 154,7 | f773f5bc |
| limit 100 MiB | 35 | 99,98 | 152,8 | a8939715 |
| limit 100 MiB + `ffn_` | 13 | 95,98 | 145,3 | 35969901 |
| minta, ami nem illeszkedik (`zzz`) | 0 | (nincs) | 123,8 | 2be92339 (= repack) |

A prompt-sebesség a duplikált súlyok arányával nő, 0 duplikált súlynál a viselkedés a `repack`-ével azonos (md5 `2be92339`). Az `ffn_` tenzorok adják a repackelt súlyok nagy részét (265,78 / 308,18 MiB), tehát RAM-szűkös esetben a `--dual-buffer-pattern ffn_` a legjobb ár-érték.

A 4. fázis után az alapviselkedés újra ellenőrizve (ugyanaz a bináris): repack `2be92339`, host `0063c4c5`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual_noopoff `2be92339`, dual `35969901`: mind egyezik a korábbiakkal.

## Figyelmeztetés a mérési módszerhez
A kimenet md5-jének összehasonlításakor a stderr-t **fájlba** kell irányítani (ahogy a `bench.sh` teszi). Ha a stderr `/dev/null`-ra (Windowson NUL) megy, a program ANSI színkódokat ír a kimenetbe (+9 bájt), és az md5 eltér, miközben a modell kimenete azonos. Ezt egy hibás kézi ellenőrzés közben vettem észre.

## v2: host-puffer a nem repackelhető tenzoroknak + RAM-védelem
Laptop, Qwen2.5-1.5B, `-ngl 14`, a bench.sh teljes sorozata (3 ismétlés).
- **Elfogadási feltételek:** minden mód md5-je azonos a korábbiakkal (rövid `8e2d6759` mindenhol; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`). A `dual_noopoff` = repack kimenet, a dual prompt a host szintjén: hosszú dual 330,0 (322,9-333,7) vs host 317,2 vs repack 122,6 tok/s; rövid dual 214,2 (153,4-261,5; nagy szórás) vs host 238,5. Generálás: dual 19,8, repack 19,2, host 18,7 (laptopon nem mérvadó).
- **Pufferek dual módban:** `CPU_REPACK=308,18 Vulkan0=526,55 Vulkan_Host=533,33` - sima `CPU` puffer nincs (korábban `CPU=225,15`), a `Vulkan_Host` mérete megegyezik a host módéval (533,33), vagyis a 225 MiB nem repackelhető súly most a host-pufferben van.
- **RAM-védelem** (`LLAMA_DUAL_BUFFER_FREE_MIB` a szabad RAM felülírására, mert a gépen 5,5 GiB szabad volt; tartalék 1024 MiB, CPU-s rész ~548 MiB):
  - szabad 5561 MiB: keret 3989 MiB, 87 súly duplikálva (308,18 MiB), változatlan viselkedés;
  - szabad 1700 MiB: keret 128 MiB, 44 súly duplikálva (127,62 MiB), `Vulkan_Host` 352,77 MiB;
  - szabad 1500 / 1300 MiB: keret 0, figyelmeztetés "not enough free RAM ... disabling it", normál host-puffer lista (`Vulkan_Host` 533,33, nincs `CPU_REPACK`);
  - szabad 1300 MiB + `--dual-buffer-limit 100`: a kézi limit nyer, 35 súly (99,98 MiB).
- **Nem tesztelt:** MoE modell és a NAS (ROCm); a qwen3-coder:30b prompt-különbség (438 vs 515 tok/s) javulását ott kell mérni.

## v3: cgroup-korlát
- Laptop (Windows, nincs cgroup): a naplóban `free RAM 2270 MiB (system)`, 87 súly duplikálva (308,18 MiB), `Vulkan_Host` 533,33 MiB; `--dual-buffer-limit 100` -> 35 súly (99,98 MiB). A viselkedés a v2-vel azonos.
- A `get_cgroup_free()` önálló programban, hamis cgroup-fán: v2 beágyazott (`a`: max 1000 MiB, használat 400 MiB, inaktív cache 100 MiB; `a/b` korlátlan) -> szabad 700 MiB, korlát 1000 MiB; v1 (500 MiB korlát, 200 MiB használat) -> 300 MiB; gyökér cgroup / nincs memory vezérlő / nem létező fájl -> nincs korlát.
- Nem tesztelt: valódi cgroup (Docker `--memory`) a NAS-on.
