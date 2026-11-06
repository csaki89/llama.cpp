# Összefoglaló: dual-buffer (éjszakai munka)

## Mi készült el
- **`--dual-buffer`** (`LLAMA_ARG_DUAL_BUFFER`, alapból ki): a CPU-n maradó, repackelhető súlyok két példányban vannak meg. Az elsődleges a `CPU_REPACK` pufferben (a gráf és a CPU ezt használja), mellette egy eredeti formátumú iker a GPU-backend host-pufferében (`Vulkan_Host` / `ROCm_Host`). A scheduler, ha egy műveletet a GPU-ra offloadol, az elsődleges súlyt az ikerre cseréli, és a meglévő host-puffer útvonalon másolja át. Kikapcsolva a viselkedés azonos az eredetivel.
- **Korlátozás** (4. fázis): `--dual-buffer-pattern REGEX` (tenzornév) és `--dual-buffer-limit MiB` (RAM-keret, tenzorsorrendben vág). Alapból minden repackelhető CPU-s súly duplikálódik.
- **Backend-független:** a változás a `ggml-backend` schedulerben (új API: `ggml_backend_sched_set_weight_alt`) és a modellbetöltőben él. Nincs Vulkan/ROCm-specifikus kód: a host-puffertípust a `ggml_backend_dev_host_buffer_type()` adja, az iker-feltételt puffertípus-tulajdonságok adják (nem host, CPU/ACCEL eszköz).
- **Napló:** `--log-verbosity 4` mellett (és alapból is, INFO szinten) a `load_tensors: dual-buffer: N weights also stored in <puffer> (X MiB)` sor és a pufferméretek (`CPU_REPACK`, `Vulkan_Host`).
- **Dokumentáció:** `DESIGN.md` (terv), `DECISIONS.md` (10 döntés), `PROGRESS.md`, `baseline.md` (alapmérés + NAS-összevetés), `test-results.md` (ellenőrzés). Eszközök: `tools/dualbuf/` (`bench.sh`, `modes.conf`, `summarize.sh`, `build-vulkan.bat`).
- **Commitok** a `dual-buffer` branchen, push nélkül (a push kifejezett kérésre lehet).

## Eredmény a laptopon (Qwen2.5-1.5B Q4_K_M, `-ngl 14`, 3 ismétlés)
Részletek: `test-results.md`. Minden elfogadási feltétel teljesült:
- kikapcsolva a meglévő módok md5-je és puffere egyezik a baseline-nal;
- `--dual-buffer --no-op-offload`, hosszú prompt: kimenet = repack + no-op-offload (`2be92339`), tehát a CPU a repackelt példánnyal számol;
- `--dual-buffer` op offloaddal: prompt rövid 255,7 / hosszú 304,5 tok/s, a `host` 268,2 / 307,0 és a `repack` 136,8 / 129,5 mellett (vagyis a host-puffer szintjén);
- rövid prompton mind a 8 mód kimenete bájtra azonos;
- a naplóban `CPU_REPACK` 308,18 MiB és `Vulkan_Host` 308,18 MiB; a mért host-RAM többlet 308 MiB (egyezik a várt ~308 MiB-tal).

## Ami nincs kész / nem tesztelt
- **Generálási sebesség:** a laptopon nem mérhető (az iGPU nem gyorsabb a CPU-nál, és a RAM közös); a várt nyereség (generálás a repack szintjén, prompt a host+offload szintjén) csak a NAS-on igazolható.
- **ROCm/HIP, CUDA:** nem fordítottam, nem futtattam.
- **MoE modellek (`MUL_MAT_ID`, expert tenzorok):** nem teszteltem. Ha a gráf egy súly *nézetét* kapja `src`-ként, az iker-keresés nem talál, és a viselkedés a mai marad (nincs offload) (DESIGN.md 4.).
- **Többeszközös gép**, **LoRA**, **`llama-server` hosszú futás**, **mtmd/clip** (saját schedulere van, nem érinti a változás): nem teszteltem.
- **`--fit`:** az ikertenzorok belekerülnek a host-puffer méretébe, de a `--fit` csak az eszközmemóriát ellenőrzi, a host-RAM többletet nem. Nagy modellnél (pl. a gpt-oss:120b ~57 GB CPU-s része duplán nem fér el a 128 GB-ban) a `--dual-buffer-limit` / `--dual-buffer-pattern` a kézi védelem. Automatikus RAM-keret nincs.
- **Háromzónás kiosztás**, **Ollama-integráció**: nem cél volt.
- **mmap:** a `--dual-buffer` explicit `--load-mode mmap` mellett kikapcsol (figyelmeztetéssel). Itt az `auto` mód nem mmap, ezért ott aktív; NAS-on ezt ellenőrizni kell, vagy mindig `--load-mode none`-t kell adni (a referencia-beállításokban ez benne van).

## Nyitott kérdések
1. Az iker-csere a scheduler 1. passában történik, a gráf csomópontjának `src[]`-jét módosítva (mint az 5. pass másolatai). Elfogadható-e upstreamben, vagy a `ggml_tensor`-ban lenne szebb az iker mutatója? (DESIGN.md 3., B.)
2. A scheduler hash-halmaza a gráfmérethez igazodik; az ikrek új bejegyzések. A tipikus modelleknél van tartalék, nagyon sok súlyú modellnél ellenőrizendő (`GGML_ASSERT` / hash-tele hiba esetén).
3. A `--dual-buffer-limit` tenzorsorrendben vág (korai rétegek előnyben). Jobb lehet az `ffn_` tenzorokat előnyben részesíteni (a mérés szerint az adják a repackelt tömeg nagy részét: 265,78 / 308,18 MiB).
4. A `--dual-buffer --no-host` együtt: a dual-buffer nyer (DECISIONS.md 3.). Kell-e erről külön figyelmeztetés?

## NAS-on teendő (ROCm build és mérés)
1. **Letöltés:** `git clone -b dual-buffer https://github.com/csaki89/llama.cpp.git` (vagy meglévő klónban `git fetch origin dual-buffer`). A branchet a felhasználónak kell az `origin`-re pusholnia, mert a push nem engedélyezett nekem.
2. **Build (ROCm):** a repóban lévő HIP build leírás szerint (`docs/build.md`), pl. `cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1031 -DCMAKE_BUILD_TYPE=Release` (a gfx1031-et a gfx1030 kerülőút miatt `HSA_OVERRIDE_GFX_VERSION=10.3.0`-val futtatjuk). A `-DGGML_AVX2=ON` kell, ha a `GGML_NATIVE` nem detektál.
3. **Környezet ellenőrzése:** `env | grep ^LLAMA_` legyen üres (különösen `LLAMA_ARG_NO_HOST`, `LLAMA_ARG_DUAL_BUFFER*`); a `bench.sh` törli és jelzi őket.
4. **Mérés:** a `tools/dualbuf/bench.sh` Linuxon is fut, de a `modes.conf` a laptopra van hangolva (`-ngl @NGL@` = a rétegek fele, a modellből). A NAS-os sorozathoz (30 / 18 / 6 GPU-réteg, qwen3-vl:8b) háromszor kell futtatni `NGL=30`, `NGL=18`, `NGL=6`-tal, a referencia-kapcsolókkal (`--load-mode none --flash-attn auto -b 512 -ub 512 --log-verbosity 4`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`). A `bench.sh` kapcsolói most fixen `-fa auto -b 512 -ub 512 --load-mode none`-t adnak, ezek egyeznek a referenciával; a NAS-os promptok (98 token, 300 generált token) a `bench.sh` promptjaitól (110 / 1611 token, 200 token) eltérnek, ezt a promptfájlokkal és `-n 300`-zal lehet egyeztetni (`-n` kapcsoló van). Hogy a korábbi NAS-os mérések milyen programmal készültek (`llama-server` vagy más), nem tudom; a `bench.sh` a `llama-completion`-t használja, ezért a számok összevetésekor ezt érdemes tisztázni.
5. **Várt eredmény:** generálás a `repack` szintjén, prompt a host-puffer szintjén (CLAUDE.md: +23% / +59% / +64% a repackhez képest 30 / 18 / 6 rétegnél). Ha a prompt a host szintjén van, de a generálás ott marad a host-pufferes (lassú) szinten, akkor a CPU nem az elsődleges (repack) példányt használja: ezt a naplóban a `CPU_REPACK` jelenléte és a `--dual-buffer --no-op-offload` md5-je (= repack + no-op-offload) mutatja.
6. **RAM:** a duplikált rész a repackelt CPU-s súlyok mérete; 8B Q4_K_M modellnél 30 / 18 / 6 GPU-rétegnél a CPU-s rész növekszik, ezt a naplóbeli `Vulkan_Host`/`ROCm_Host` sor mutatja.

## Folyamat közben észrevettek
- A settings.json `Bash(cmd //c tools/dualbuf/build-vulkan.bat)` szabálya nem működik, mert a `cmd` perjeles útvonalat nem ért; a használható forma `cmd //c "tools\\dualbuf\\build-vulkan.bat"`. Ezt a szabályt javítani kell.
- Az éjszaka során egyetlen parancs sem várt engedélyre. A munkamenet engedélyezési módja megengedőbb lehetett, mint a settings.json; a használt, de az allow listán nem szereplő parancsok: `sed`, `awk`, `grep`, `cat`, `cmp`, `md5sum`, `od`, `wc`, `ls`, `ctest`, `env`, `seq`, `rm -rf bench-out`.
- Md5-összehasonlításnál a stderr-t fájlba kell irányítani; `/dev/null`-ra (Windows NUL) ANSI színkódok kerülnek a kimenetbe (lásd `test-results.md`).
- A `ctest` ebből a környezetből `BAD_COMMAND`-ot ad (msys2-s cmake); a `test-arg-parser` közvetlen futtatással átmegy.
