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

## Baseline a K/V-mozgatás után (opció nélkül)
`bench.sh -r 1` (8 mód x 2 prompt, 1 ismétlés; az md5 determinisztikus), a `a6dd34d00` kódon. Mind a 16 futás md5-je és puffere egyezik a korábbiakkal: rövid `8e2d6759` mindenhol; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`. Ezzel a korábban kimaradt `dual_noopoff` hosszú eset is ellenőrizve (`2be92339`). Opció nélkül a naplóban nincs `flex` sor, a KV-pufferek a megszokottak (rétegenkénti KV-puffer csak `--flex-layers` mellett keletkezik).

## A rugalmas sáv befejezése (sáv a határon, host-példány, DMA-másolás)
Laptop (Vulkan, Iris Xe iGPU, közös memória), Qwen2.5-1.5B, `-ngl 14 --flex-layers 2` (a sáv most a **15. és 16.** réteg, a CPU-s rétegek mellett), hosszú prompt 1611 token, 200 generált token, `n_ctx` 32768, temperature 0. Szkript: `tools/dualbuf/flex-test.sh` (esetenként 1 futás; a mátrix a `ddfd52fd6`, a visszatételi idők az `aba2179ec` kódon).

### Opció nélkül
`bench.sh -r 1` az `aba2179ec` kódon: mind a 16 futás md5-je és puffere egyezik a korábbiakkal (rövid `8e2d6759` mindenhol; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`; pl. dual `CPU_REPACK=308,18 Vulkan0=526,55 Vulkan_Host=533,33`). A dual-buffer viselkedése nem változott; a keret naplósorában új a `flex copies 0 MiB` mező.

### Mátrix
"VRAM" = a futás végi `common_memory_breakdown_print` Vulkan0 sora (self = model + K/V + compute; a compute mindenhol 308). "Hasítás" = a `sched_reserve` gráfjai (prompt / generálás) a mozgatás után.

| eset | prompt tok/s | gen tok/s | md5 | VRAM self (model + K/V) | hasítás pp / tg |
|---|---|---|---|---|---|
| flex, mozgatás nélkül | 220,2* | 11,4 | 0063c4c5 (= host) | 1251 (526 + 416) | 212 / 2 |
| 1 levéve, host-példánnyal | 260,1 | 15,5 | 3721153f | 1194 (501 + 384) | 226 / 2 |
| 1 levéve, host-példány nélkül (`LLAMA_FLEX_TEST_HOST=0`) | 220,1 | 13,0 | 3721153f | 1194 (501 + 384) | 224 / 2 |
| 2 levéve, host-példánnyal | 252,6 | 14,6 | c2356003 | 1133 (472 + 352) | 240 / 2 |
| 2 levéve, host-példány nélkül | 215,6 | 15,4 | 35969901 | 1133 (472 + 352) | 237 / 2 |
| 2 le +19, vissza +39 tokennél (host-példányból) | 267,6 | 15,2 | 0063c4c5 | 1251 (526 + 416) | 212 / 2 -> 240 / 2 -> 212 / 2 |
| ugyanez, visszatétel a fájlból | 266,2 | 15,2 | 0063c4c5 | 1251 (526 + 416) | ugyanaz |
| prompt a GPU-n, 2 levéve a generálásra | 273,7 | 15,4 | a8939715 | 1133 (472 + 352) | 212 / 2 -> 240 / 2 |
| `-ngl 14 --dual-buffer` (referencia) | 271,9 | 15,0 | 35969901 | 1251 (526 + 416) | 212 / 2 |
| `--dual-buffer`, flex, mozgatás nélkül | 271,7 | 16,1 | **35969901** | 1251 (526 + 416) | 212 / 2 |
| `-ngl 13 --dual-buffer` (referencia: a 15. réteg CPU-s) | 248,3 | 13,3 | 2e66983a | 1194 (501 + 384) | 226 / 2 |
| `--dual-buffer`, flex, 1 levéve | 263,1 | 13,5 | **2e66983a** | **1194 (501 + 384)** | **226 / 2** |
| `-ngl 12 --dual-buffer` (referencia: a 15-16. réteg CPU-s) | 254,6 | 15,4 | 8934babb | 1133 (472 + 352) | 240 / 2 |
| `--dual-buffer`, flex, 2 levéve | 256,8 | 17,1 | **8934babb** | **1133 (472 + 352)** | **240 / 2** |

\* az első futás, bemelegítés nélkül; a többi prompt-érték után nem összevethető (a `bench.sh`-ban a mozgatás nélküli flex 241,7, illetve 285-298 volt).

- **Egy levett réteg pontosan úgy viselkedik, mint egy eleve CPU-s dual-bufferes réteg:** 1 és 2 levett réteggel a kimenet bájtra, a VRAM-foglalás és a gráfhasítások száma egyezik a `-ngl 13` / `-ngl 12 --dual-buffer` futással.
- **Gráfhasítások** (generálás) 0, 1, 2 levett réteggel: **2 / 2 / 2** (a sáv középről levett rétegeivel korábban 14 / 10). A prompt-gráf 212 / 226 / 240, ugyanannyi, mint a referenciáké (a CPU-s rétegek op offloadja rétegenként +14 hasítás).
- **Felszabaduló VRAM rétegenként** (pufferelszámolás): a 15. réteg 57 MiB (25 súly + 32 K/V), a 16. réteg 61 MiB (29 + 32); kettő együtt 118 MiB.
- **Prompt-sebesség levett rétegekkel:** a host-példány 1 levett rétegnél 220 -> 260, 2-nél 216 -> 253 tok/s; a referenciák (eleve CPU-s dual-bufferes réteg) 248 / 255. A laptopon a szórás nagy (lásd a csillagot), a generálás itt nem mérvadó.
- **Kimenet:** minden esetben értelmes. Ahol a réteg a prompt és a generálás alatt ugyanott van, bájtra egyezik a megfelelő eredetivel (mozgatás nélkül a host/dual alapfutással, végig levéve az eleve CPU-s dual-bufferes futással; a le-vissza ciklus után is az eredetivel). Ahol a prompt és a generálás más elhelyezésben fut, az első eltérés a 397-660. bájtnál van, a szöveg értelmes.

### Másolási idők
**Visszatétel (súlyok)**, rétegenként 25-29 MiB, `-c 32768`:

| forrás | idő (foglalás + másolás) |
|---|---|
| host-példány + CPU-iker, szinkron `set` (`aba2179ec`) | 6,1-9,7 ms (0,6-1,9 + 4,8-8,6), ~3,5-5,5 GB/s |
| host-példány, aszinkron `set` (`ddfd52fd6`) | 8,8-15,9 ms |
| modellfájl (tartalék, meleg page cache) | 19,6-41,2 ms |

**Levétel (súlyok):** 0,2-2,4 ms (mező-csere + a GPU-puffer felszabadítása).

**K/V (rétegenként), szétbontva** (alloc / clear / copy+sync / free, ms). A prompt után 1628 (levételkor) / 1648 (visszatételkor) cella használt. A szinkronizálás a mozgatás előtt minden esetben 0,0 ms (a decode végén már megtörtént).

| irány | kontextus | előtte (`LLAMA_FLEX_TEST_COPY_LEGACY=1`: CPU-puffer, teljes szinkron másolás) | utána (rögzített host-puffer, csak a használt cellák, DMA) |
|---|---|---|---|
| GPU -> CPU | 4096 (4 MiB) | 17,0-17,5 = 0 / 0,8 / 16,1-16,6 / 0,2 | 1,7-2,3 = 0,4 / 0,5 / 0,5-1,3 / 0-0,3 |
| GPU -> CPU | 32768 (32 MiB) | 143-155 = 0 / 9,6-11,1 / 134-144 / 0,2-0,3 | 9,9-20,5 = 6,7-17,6 / 2,2-2,4 / 0,5-0,6 / 0-0,5 |
| CPU -> GPU | 4096 | 1,4-1,5 = 0,3-0,6 / 0,2 / 0,3-0,5 / 0,4-0,5 | 1,4 = 0,7-1,1 / 0,1-0,6 / 0,2 / 0 |
| CPU -> GPU | 32768 | 8,7-63,5 = 1,0-51,7 / 1,1-1,5 / 2,6-5,9 / 4,0-4,3 | 5,7-34,5 = 1,1-31,5 / 2,1-4,1 / 0,2-0,8 / 0-0,2 |

- **Az ok** (GPU -> CPU): a Vulkan backend UMA-eszközön a szinkron olvasást a leképezett, gyorsítótár nélküli eszközmemóriából `memcpy`-vel végzi (~0,23 GB/s: 32 MiB 134-144 ms alatt). Az aszinkron olvasás rögzített memóriába a GPU másolóparancsát használja: 1,59 MiB 0,5 ms alatt (DECISIONS.md 19-20).
- **A maradék** a foglalás: rögzített host-memória 32 MiB-ra 5-18 ms, Vulkan eszközmemória 1-52 ms (erősen szór). A másolás maga 0,2-1,3 ms. Ezt a hardver sebességére a pufferek újrafoglalásának elkerülése hozná le (Következő lépések, DESIGN-3zone.md).
- A kimenet a régi és az új úton is bájtra azonos (`964b3996`, 60 token, mindkét kontextusmérettel).

### Nem tesztelt
ROCm/HIP (HIP-gráfok a címváltozás után, PCIe-sebesség), dedikált GPU eszközmemória-lekérdezése, MoE, több context, iSWA/hibrid memória, `llama-server`, transzponált V (flash attention nélkül), a RAM-védelem kiváltása (`flex_no_host`, a sáv kikapcsolása) valódi RAM-hiánnyal.

## Automatikus háromzónás működés (`--auto-zones`)
Laptop (Vulkan, Iris Xe iGPU), Qwen2.5-1.5B, `-c 32768`, `-fa auto -b 512 -ub 512`, temperature 0. A sáv kényszerítése a `--fit-target`-tel (`-fitt 6500`: a fit 6,5 GiB tartalékot hagy), a keret a `LLAMA_ZONES_HEADROOM_MIB`-bel. Szkriptek: `tools/dualbuf/zones-test.sh` (llama-server), a sebességsor kézi `llama-completion` futás.

### Kikapcsolva
`bench.sh -r 1` a `047a72477` kódon: mind a 16 futás md5-je és puffere egyezik a korábbiakkal (rövid `8e2d6759`; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`; pl. dual `CPU_REPACK=308,18 Vulkan0=526,55 Vulkan_Host=533,33`). A dual-buffer viselkedése nem változott.

### A zónák (1. rész)
| `-fitt` (MiB) | 1. zóna (`n_ctx` 32768) | 1. + 2. zóna (`n_ctx` 4096) | sáv |
|---|---|---|---|
| 1024 | minden réteg | - | 0 (minden elfér a max kontextussal) |
| 6300 | 11 | 20 | 9 (9-17. réteg) |
| 6500 | 7 | 13 | 6 (16-21.) |
| 6700 | 4 | 7 | 3 (22-24.) |
`--zones-ctx-init 512` mellett (`-fitt 6500`): `llama-completion` 7 / 15, `llama-server` 10 / 23 réteg (a szerver compute-puffere más).

### Dinamikus KV (2. rész), statikus elhelyezéssel (`-ngl 14`, `--zones-ctx-init 512`)
- 512 cellával indul, a darab 2560 cella (70 MiB az összes rétegre). Hosszú prompt (1611 token): egy növelés 512 -> 2560 (66 ms: alloc 57, copy 9), a VRAM-ban a KV 416 helyett **32 MiB**; a kimenet **bájtra azonos** a statikus KV-s futással (`0063c4c5`), `-b 2048`-cal is.
- Javított mérési hiba: a decode elején hívott `synchronize()` a sorba állított adagot ~0 ms-mal számolta el (DECISIONS.md 23).

### Zónaváltás egy beszélgetésben (3. rész), `llama-server`
`LLAMA_ZONES_HEADROOM_MIB=30`, `--auto-zones -fitt 6500 --zones-ctx-init 512 -c 32768`; sáv 13 réteg (6-18.). Kérések egymás után, ugyanazon a sloton:

| kérés | prompt / gen token | prompt tok/s | gen tok/s | ami történt |
|---|---|---|---|---|
| hosszú (long1) | 6414 / 100 | 200,7 | 12,66 | KV 512 -> 2560 -> 5120 -> 7680; közben a 6., 7., 8., 9. réteg le (a határ felől) |
| új, rövid (short) | 10 / 100 | 26,3 | 19,23 | KV 7680 -> 2560 (üres), a 9., 8., 7. réteg vissza (fordított sorrend), a 6. a keret miatt nem |
| a hosszú újra (long2) | 1 / 100 (a szerver RAM-os prompt-gyorsítótárából) | - | 12,80 | állapotbetöltés: KV 2560 -> 7680, utána a 7., 8., 9. réteg le (könyvelt keret) |

A long1 és long2 kimenete azonos (`10476d2c`), a szöveg minden kérésnél értelmes.

**A mozgatások ideje (a foglalással együtt):**
| művelet | idő | ebből foglalás |
|---|---|---|
| levétel: súlyok | 0,0-1,4 ms | - |
| levétel: K/V (0,5-7,5 MiB, a használt cellák DMA-val) | 1,0-3,3 ms | 0,2-1,1 ms |
| visszatétel: súlyok (28,6 MiB, host-példányból) | 3,6-28,6 ms | 1,2-26,1 ms |
| visszatétel: K/V (üres KV) | 0,3-0,4 ms | 0,2-0,3 ms |
| KV növelés (az összes réteg, 14-140 MiB másolva) | 30-94 ms | 17-78 ms |
| KV csökkentés üres KV-nál | 15 ms | 9,5 ms |
A foglalás továbbra is a nagyobb rész (a Vulkan-eszközmemória foglalása 1-78 ms, erősen szór); a host-oldali K/V-puffer újrahasznosítása ebben a sorban nem jött létre, mert a KV a két levétel között nőtt (a régi puffer kicsi volt).

### Bájtos egyezés a statikus elhelyezéssel
Ahol a rétegek helye a kérés alatt nem változik: `LLAMA_ZONES_HEADROOM_MIB=-50 --auto-zones -fitt 6500 --dual-buffer` (13 GPU-réteg, a sávból 2 réteg a kérés előtt le) kimenete `f773f5bc`, azonos a statikus `-ngl 11 --dual-buffer` futással (`f773f5bc`). A KV-növelés a kimenetet nem változtatja (a 2. rész fenti mérése).

### Generálási sebesség a kontextus hosszával
`llama-completion`, 100 generált token. "statikus 7" = az 1. zóna (ami a teljes 32K KV-vel elfér), "statikus 13" = 1. + 2. zóna teljes 32K KV-vel (a valódi kereten túl; a laptopon a közös memória miatt elfér), "auto" = `--auto-zones -fitt 6500`, `LLAMA_ZONES_HEADROOM_MIB=60`.

| prompt | eset | prompt tok/s | gen tok/s | md5 | VRAM: model + K/V (MiB) |
|---|---|---|---|---|---|
| 1611 | statikus 7 | 273,7 | 18,11 | 42a75edb | 340 + 192 |
| 1611 | statikus 13 | 302,7 | 18,58 | 94748168 | 501 + 384 |
| 1611 | auto (KV 4096, nincs mozgatás) | 308,4 | 17,78 | 94748168 | 501 + 48 |
| 6414 | statikus 7 | 185,8 | 11,02 | ddd5ed9c | 340 + 192 |
| 6414 | statikus 13 | 192,7 | 12,13 | ddd5ed9c | 501 + 384 |
| 6414 | auto (KV 4096 -> 5120 -> 7680, nincs mozgatás) | 193,8 | 11,83 | ddd5ed9c | 501 + 90 |

- Az auto a statikus 13 sebességén fut, az 1. zóna VRAM-jánál alig többel a KV-ben (48-90 MiB a 384 helyett); ebben a sorban a 60 MiB-os keret elég volt, így réteg nem mozgott (a mozgatással járó sebességet a szerveres sor mutatja: 12,7 tok/s 4 levett réteggel 6,4k kontextusnál, 19,2 visszatétel után rövid kontextusnál).
- A laptopon a különbségek kicsik (iGPU, közös memória); a hatást a NAS-on kell mérni (SUMMARY.md 7.).
- **Nem futott le:** a 12,8k tokenes sor (a `statikus 7` futás közben a rendszer kevés memória miatt leállította a mérést; nem indítottam újra).

### Nem tesztelt
VMM (CUDA/HIP, nem készült el), ROCm/HIP-gráfok a címváltozások után, dedikált GPU-n a keret pontossága, MoE (az automatikus sáv ott kimarad), több slot/sorozat (`-np` > 1), transzponált V (flash attention nélkül a dinamikus KV kikapcsol), a 12,8k-s sebességsor.

## Valódi használat: MoE, VMM, Flash Attention (`e86f222a5`)
Laptop (Vulkan, Iris Xe iGPU). Modellek: Qwen2.5-1.5B (dense) és **granite-3.1-1b-a400m-instruct Q4_K_M** (MoE, 24 réteg, 32 szakértő, 784 MiB; nyilvános GGUF, bartowski).

### Kikapcsolva
`bench.sh -r 1`: mind a 16 futás md5-je és puffere egyezik a korábbiakkal (rövid `8e2d6759`; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`; dual `CPU_REPACK=308,18 Vulkan0=526,55 Vulkan_Host=533,33`). A dual-buffer változatlan.

### MoE: a zónák a két fit különbségéből
| beállítás | 1. zóna (max kontextus) | 1. + 2. zóna (kezdő kontextus) | sáv |
|---|---|---|---|
| `-c 32768 -fitt 6600` | 10 GPU-réteg, 9 override | 25 (minden), 12 override | 15 egész réteg (0-14.; 422 MiB a CPU-ikrekben) |
| `-c 12288 -fitt 6500 --zones-ctx-init 1024` | 25 (minden dense), 22 szakértő-override | 25, 3 override | **részleges**: a 3-22. réteg szakértői (542 MiB) |
| `-c 16384 -fitt 6500` | 22 GPU-réteg, 21 override | 25, 3 override | 3 egész + részleges (623 MiB) |
A fit MoE-nél a dense részt is visszaveheti (32K-n a K/V ~1,5 GiB), ekkor egész rétegek is a sávba kerülnek; a különbség-szabály mindkettőt kezeli.

**Szerveres beszélgetés MoE-vel** (`-c 12288 --auto-zones -fitt 6500 --zones-ctx-init 1024`, `LLAMA_ZONES_HEADROOM_MIB=20`): a hosszú kérés (7031 token) alatt a KV 1024 -> 7680 cellára nőtt, a szakértő-rétegek **csökkenő sorrendben** (23, 22, ..., 14.) kerültek le (rétegenként 27-31 MiB, levétel 0-3,6 ms; a réteg figyelemszámítása és K/V-je a GPU-n maradt); az új rövid kérésnél a KV 7680 -> 1536-ra csökkent, és a rétegek **növekvő sorrendben** vissza (2,5-31 ms rétegenként, a host-példányból); a hosszú kérés újra (RAM-os prompt-gyorsítótár) után a KV megnőtt, és a rétegek a keret szerint újra lekerültek. A hosszú kérés kimenete mindkétszer azonos (`f3980366`), a szöveg minden kérésnél értelmes. Az elfogadási futásban 30 mozgatás.

**Bájtos egyezés MoE-vel:** `--auto-zones --dual-buffer` (`LLAMA_ZONES_HEADROOM_MIB=-30`, a kérés előtt 2 szakértő-réteg le) és a statikus `-fit off -ngl 25 --dual-buffer -ot ...=CPU` kimenete azonos (`4020432f`); ennél a modellnél és promptnál azonban a statikus változat a két réteg levétele nélkül is ugyanezt adja, tehát ez az összevetés itt nem érzékeny (a dense modellen az egyezés korábban valódi különbséget is mutatott: `f773f5bc`).

### VMM (növelhető KV-pufferek)
- A laptopos Vulkan-backend nem ad VMM-függvényt: `set_dynamic: ... growth on the device: 0/22 layers in place (VMM), 22 by copy (no VMM on this backend/device, or flex layers)`, a tartalék út fut (változatlan kimenet).
- A helyben növelés logikája a `LLAMA_KV_VMM_SIM=1` szimulációval: `-ngl 14 --auto-zones --zones-ctx-init 512 -c 32768`, hosszú prompt: `set_dynamic: ... 13/13 layers in place (VMM)`, a növelés `relayout: KV 512 -> 2560 cells (... layers: 13 grown in place, 0 new growable, 15 new by copy), 18,9 ms` (a 15 CPU-s réteg másolással); a kimenet **bájtra azonos** a statikus KV-s futással (`0063c4c5`), ahogy a tartalék úté is.
- A valódi CUDA/HIP VMM-kód (`ggml-cuda.cu`) a laptopon nem fordítható; a ROCm-fordítás az ollama-dualbuf CI-jén fut.

### Flash Attention
A napló minden dinamikus KV-s futásnál: `zones: flash attention on, dynamic KV on` (Vulkan, `-fa auto`). A kikapcsolt FA (`-fa off`) esetén a `set_dynamic` figyelmeztet (`dynamic KV disabled: the V cache is transposed because flash attention is off`), és a teljes KV foglalódik.

### Szerveres beszélgetés (dense), változatlan
`-c 32768 --auto-zones -fitt 6500 --zones-ctx-init 512`, `LLAMA_ZONES_HEADROOM_MIB=30`: long1 / short / long2 kimenete `10476d2c` / `d640dab1` / `10476d2c`, mint korábban; a rétegek (6-9.) a KV-vel le, új beszélgetésnél (9., 8., 7.) vissza.

### Nem tesztelt
Valódi VMM (CUDA/HIP), ROCm, nagy MoE (qwen3-coder:30b), több slot. A megszakadt 12,8k-s sebességsort nem ismételtem meg (kérés szerint).

### VMM aréna (DECISIONS 32), laptop
A változás a ggml-cuda-ban és a KV relayoutban van; a laptopos (Vulkan) út nem kap arénát. Ellenőrzés: dinamikus KV a tartalék úton `0063c4c5` (mint a statikus), `LLAMA_KV_VMM_SIM=1`-gyel (az aréna-API szimulációja) `set_dynamic: ... 13/13 layers in place (VMM)`, a növelés `13 grown in place`, a kimenet `0063c4c5`; szerveres beszélgetés-próba `10476d2c` / `d640dab1` / `10476d2c`, mint korábban.

### VMM: leképezés-várakozás és commit (DECISIONS 33-35), laptop
A valódi VMM-út a laptopon nem fut (nincs ROCm). Ellenőrzés: `bench.sh -r 1` mind a 16 md5-je és puffere a korábbi; dinamikus KV a tartalék úton és `LLAMA_KV_VMM_SIM=1`-gyel `0063c4c5`; szerveres próba dense `10476d2c` / `d640dab1` / `10476d2c` (20 mozgatás), MoE `f3980366` / `70b4fb2a` / `f3980366` (30 mozgatás).

## HIP: csak a korlát, másolásos tartalék a commit hibájánál (DECISIONS.md 40)

`bench.sh -r 1`: mind a 16 futás md5-je és puffere a korábbi (rövid `8e2d6759`; hosszú: repack/repack_noopoff/dual_noopoff `2be92339`, host `0063c4c5`, host_noopoff `df42ead7`, cpu_only `f773f5bc`, gpu_full `a8939715`, dual `35969901`). Szerveres próba (dense, `LLAMA_ZONES_HEADROOM_MIB=30`, `-c 32768 --auto-zones -fitt 6500 --zones-ctx-init 512`): `LLAMA_KV_VMM_SIM` 0, 1, 2 és 3 mellett egyaránt `10476d2c` / `d640dab1` / `10476d2c`, 20 mozgatás. A 2-es szinten (minden commit hibát ad) 6 `could not be committed` hibasor, minden réteg közönséges pufferrel és másolással nő; a 3-as szinten (csak a helyben növelés commitja hibázik) 4 hibasor, a rétegek új szakaszba költöznek (`0 grown in place, 9 new growable, 19 new by copy`). MoE (granite, `-c 12288 ... --zones-ctx-init 1024`, `HEADROOM=20`): `f3980366` / `70b4fb2a` / `f3980366`, 30 mozgatás. A HIP-kód a laptopon nem fordul (a NAS-on a CI image-ében).
