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

## v4: a CPU-s rész becslése az elhelyezésből
Laptop, Qwen2.5-1.5B, `LLAMA_DUAL_BUFFER_FREE_MIB=6000`. "becslés" = a naplóbeli `CPU part`, "tényleges" = a pufferek összege a RAM-ban (CPU_REPACK + host-puffer).
| beállítás | becslés (MiB) | tényleges (MiB) |
|---|---|---|
| `-ngl 14` | 716 | 533 (CPU_REPACK 308 + 225 nem repackelhető) - óvatos |
| `-ngl 0` | 1060 | 1060 |
| nincs `-ngl` (minden GPU-n, fit) | 125 | 125 (`token_embd`) |
| `-ngl 99 -ot "ffn_.*=CPU"` | 793 | 793 |
| `-ngl 99 -ot "blk\.1[0-9]\..*=CPU"` | 390 | 390 |
| `-ngl 99 -ot "ffn_.*=Vulkan0"` | 125 | 125 |
A javítás előtti bináris `-ngl 99 -ot` mellett `CPU part ~0 MiB`-ot adott (a hiba reprodukálva). A NAS-os Ollama + MoE esetet ott kell újramérni.

## Háromzónás elhelyezés, 1. építőkocka: `--flex-layers` (rétegek áthelyezése futás közben)
Laptop (Vulkan, Iris Xe iGPU), Qwen2.5-1.5B (28 réteg), `-ngl 14`, `--flex-layers 2` (a 26. és 27. réteg), hosszú prompt (1611 token), 200 generált token, temperature 0. Szkript: `tools/dualbuf/flex-test.sh` (2 ismétlés, a dual-buffer esetek 1). Kiváltás: `LLAMA_FLEX_TEST="<token>:<take|put>:<réteg>,..."`.

### Opció nélkül (baseline)
`bench.sh`, 8 mód x 2 prompt x 3 ismétlés. Az utolsó 3 futás (`dual_noopoff`, hosszú prompt) nem futott le, mert a rendszer kevés memória miatt leállította a mérést; nem indítottam újra. A lefutott 45 futásban minden md5 és puffer egyezik a korábbiakkal: rövid `8e2d6759` mindenhol; hosszú: repack/repack_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`; pufferek pl. repack `CPU=225,15 CPU_REPACK=308,18 Vulkan0=526,55`, host `Vulkan0=526,55 Vulkan_Host=533,33`, dual `CPU_REPACK=308,18 Vulkan0=526,55 Vulkan_Host=533,33`. Opció nélkül a naplóban nincs `flex` sor.

### Pufferek `--flex-layers 2` mellett
`Vulkan0=472,83` (a többi GPU-s réteg) + `Vulkan0=25,12` (26. réteg) + `Vulkan0=28,60` (27. réteg) = 526,55, mint opció nélkül; a CPU-s ikrek: `CPU_REPACK=42,61` + `CPU=11,11` (a nem repackelhető normák stb.) = 53,72 MiB (`flex layers 26-27: 24 weights with a CPU copy (53.72 MiB)`). A `Vulkan_Host` 533,33 változatlan. Mozgatás nélkül a kimenet azonos az opció nélkülivel (`0063c4c5`).

### Eredmények
| eset | prompt tok/s (r1 / r2) | gen tok/s (r1 / r2) | md5 | VRAM a futás végén (modell, MiB) |
|---|---|---|---|---|
| opció nélkül | 279,8 / 301,2 | 11,6 / 16,4 | 0063c4c5 | 526 |
| flex, mozgatás nélkül | 285,3 / 297,9 | 15,3 / 16,6 | 0063c4c5 | 526 |
| 26+27 végig levéve (`0:take`) | 208,7 / 252,3 | 11,4 / 16,1 | 0063c4c5 | **472** |
| levéve a promptra, vissza a generálásra | 199,0 / 259,8 | 14,6 / 16,3 | 0063c4c5 | 526 |
| prompt a GPU-n, levéve a generálásra | 236,9 / 301,4 | 13,6 / 16,8 | 35969901 | 472 |
| ugyanez, scheduler-újrafoglalással | 207,8 / 294,9 | 14,8 / 17,1 | 35969901 | 472 |
| 27. le +50 tokennél, vissza +100-nál | 238,1 / 297,2 | 14,2 / 16,3 | 0063c4c5 | 526 |
| `--dual-buffer` | 288,3 | 16,6 | 35969901 | 526 |
| `--dual-buffer` + 27. le/vissza | 294,0 | 16,5 | 35969901 | 526 |
| `--dual-buffer` + 26+27 végig levéve | 245,5 | 15,6 | f773f5bc | 472 |

- **VRAM:** a levétel után a modell VRAM-foglalása 526 -> 472 MiB (a `common_memory_breakdown_print` Vulkan0 "model" oszlopa), a két réteg mérete 53,72 MiB. Visszatétel után újra 526. A Vulkan által jelentett szabad eszközmemória az iGPU-n nem mozdul (megosztott memória, a budget nem követi), ezért a pufferelszámolás a mérvadó; dedikált GPU-n (NAS) az eszközmemóriát is ellenőrizni kell.
- **A számítás tényleg áthelyeződik:** `GGML_SCHED_DEBUG=2` mellett a levétel után a `Qcur-27`, `ffn_up-27`, `ffn_out-27` csomópontok a CPU-n, a 26. réteg csomópontjai a Vulkanon futnak.
- **Kimenet:** minden esetben értelmes, a generálás folytatódik. Ha a réteg a prompt és a generálás alatt ugyanott van (végig levéve, vagy le-fel 50 token alatt), a kimenet bájtra azonos az eredetivel. Ha a prompt a GPU-n, a generálás a CPU-n számol (ugyanaz a helyzet, mint a dual-buffernél), a kimenet `35969901`, az első eltérés a 421. bájtnál, ugyanúgy, mint a `dual` vs `host` esetben. A `dual + végig levéve` `f773f5bc`, az első eltérés a 32. bájtnál; a szöveg értelmes ("The text is a list of sections, each containing the same sentence with slight variations...").
- **Sebesség:** levett rétegekkel a prompt lassabb (209-252 vs 280-301 tok/s), mert a repackelt tartalékot az op offload nem tudja átküldeni; visszatétel után (`levéve a promptra, vissza a generálásra`) a generálás az eredeti szintjén van (14,6 / 16,3 vs 11,6 / 16,4). A laptopon a generálás szórása nagy (r1 és r2 között 11,6 -> 16,4 az opció nélküli esetben is), ezért a generálási különbségek itt nem mérvadók.
- **Időtartam:** levétel rétegenként 0,6-3,2 ms (mező-csere és a puffer felszabadítása; a második réteg ugyanabban a lépésben <0,1 ms). Visszatétel rétegenként 20-67 ms (25-29 MiB a modellfájlból, meleg page cache mellett, foglalás + olvasás + feltöltés), kb. 0,5-1,4 GB/s. Hideg cache-sel lassabb lehet.
- **Scheduler:** a gráf újraépítése (`gf_res_prev_active = nullptr`) elég; a CPU-s compute-puffert a `ggml_gallocr` szükség esetén magától átméretezi. A `LLAMA_FLEX_TEST_RESERVE=1` (teljes újrafoglalás, ~190 ms) ugyanazt a kimenetet adja (`35969901`), ezért alapból nem fut.
- **Nem tesztelt:** ROCm/CUDA (backend-gráfok címváltozása), dedikált GPU eszközmemória-lekérdezése, MoE, több context egy modellen, hideg page cache, `llama-server`.

## Flex rétegek: a K/V követi a réteget
Laptop, ugyanaz a beállítás (`-ngl 14`, `--flex-layers 2`, hosszú prompt 1611 token, 200 generált token, `n_ctx` = 32768, f16 K/V). Szkript: `tools/dualbuf/flex-test.sh` (1 ismétlés esetenként). A flex rétegek K/V-je saját pufferben van: `Vulkan0 KV buffer size = 32.00 MiB (flex layer 26)` és `(flex layer 27)`, a többi GPU-s réteg K/V-je 352 MiB, a CPU-s rétegeké 480 MiB (összesen 896 MiB, mint opció nélkül).

"VRAM" = a futás végi `common_memory_breakdown_print` Vulkan0 sora: self = model (súlyok) + context (K/V) + compute. "Hasítások" = a ténylegesen számolt gráfok hasításszáma (`sched_graph_realloc`), prompt / generálás.

| eset | prompt / gen tok/s | md5 | VRAM self (model + K/V) MiB | hasítások prompt / gen |
|---|---|---|---|---|
| opció nélkül | 238,6 / 11,5 | 0063c4c5 | 1251 (526 + 416) | 212 / 2 |
| flex, mozgatás nélkül | 241,7 / 11,6 | 0063c4c5 | 1251 (526 + 416) | 212 / 2 |
| 26+27 végig levéve, K/V helyben (`LLAMA_FLEX_TEST_KV=0`) | 213,9 / 13,7 | 0063c4c5 | 1197 (472 + 416) | 233 / **14** |
| 26+27 végig levéve, K/V is | 192,7 / 14,8 | 0063c4c5 | **1133 (472 + 352)** | 233 / **10** |
| prompt GPU, levéve a generálásra, K/V helyben | 260,0 / 13,8 | 35969901 | 1197 (472 + 416) | 212 / 14 |
| prompt GPU, levéve a generálásra, K/V is | 243,6 / 13,3 | 35969901 | 1133 (472 + 352) | 212 / 10 |
| 27. le +50, vissza +100 tokennél | 247,3 / 13,8 | 0063c4c5 | 1251 (526 + 416) | 212 / 2 |
| levéve a promptra, vissza a generálásra | 224,2 / 15,1 | 0063c4c5 | 1251 (526 + 416) | 233 / 2 |
| `--dual-buffer`, 27. le +50, vissza +100 | 257,4 / 14,6 | 4d6570f3 | 1251 (526 + 416) | 212 / 2 |

- **Felszabaduló VRAM:** két levett réteggel K/V nélkül 54 MiB (csak a súlyok), K/V-vel együtt **118 MiB** (53,72 MiB súly + 2 x 32 MiB K/V). A K/V a CPU-s pufferbe kerül (a CPU-s rétegek K/V-je mellé), visszatétel után a VRAM újra 1251 (526 + 416).
- **A K/V átmásolása** (32 MiB rétegenként, a teljes lefoglalt K/V, 32768 cella): GPU -> CPU 148-234 ms, CPU -> GPU 11-18 ms. A visszaolvasás az iGPU-n (Vulkan) lassú; dedikált GPU-n (PCIe) mérendő. A súlyok mozgatása változatlan: levétel 0,2-3,2 ms, visszatétel 21-29 ms rétegenként.
- **Hasítások:** a generálási gráf levett rétegekkel K/V nélkül 14, K/V-vel 10 hasítást ad (opció nélkül 2); a teljes újrafoglalás (`LLAMA_FLEX_TEST_RESERVE=1`) ugyanezt mutatja (tg 14 -> 10). A prompt-gráf 233 marad, mert ott a CPU-s rétegek host-pufferes súlyai op offloaddal mennek a GPU-ra, és ezek adják a hasítások nagy részét. A maradék 10-ből rétegenként 4 abból jön, hogy a scheduler a súly nélküli műveleteket (a maradék-összeadás, `ADD`/`RMS_NORM`) a bemenetük eszközére teszi, és a reziduális út a 25. (GPU-s) rétegtől a GPU-n van: `CPU (figyelem) -> Vulkan (ADD) -> CPU (FFN) -> Vulkan (ADD)` (`GGML_SCHED_DEBUG=2`). Ez a scheduler alapviselkedése egy GPU-s réteg mögötti CPU-s rétegnél, nem a flex sajátja; a K/V mozgatása a figyelemszámítás miatti váltásokat szünteti meg.
- **Kimenet:** minden esetben értelmes, a generálás folytatódik. A K/V mozgatása a kimenetet nem változtatja meg ott, ahol a réteg a prompt és a generálás alatt ugyanott van (`0063c4c5`), és a "prompt GPU, generálás CPU" eset is ugyanazt adja K/V-vel és nélküle (`35969901`, az első eltérés az eredetitől a 421. bájtnál). A `--dual-buffer` + le/vissza eset `4d6570f3` (korábban, K/V nélkül `35969901`), az első eltérés a 421. bájtnál, a szöveg értelmes.
- **Sebesség:** a laptopon nem mérvadó (iGPU, nagy szórás); a levett rétegekkel a generálás nem lassult (13,3-14,8 vs 11,5 tok/s opció nélkül).
- **Nem tesztelt:** ROCm/HIP (HIP-gráfok, lásd DESIGN-3zone.md 4.), dedikált GPU, iSWA/hibrid memória, több sorozat (`n_stream` > 1).
