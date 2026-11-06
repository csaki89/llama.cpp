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
Cél: RX 6700 (gfx1031), TrueNAS, Docker. A build a `tools/dualbuf/Dockerfile`-lal és a `build-nas.sh`-val megy; a kártya gfx1031, de a kódot **gfx1030**-ra fordítjuk, és futtatáskor `HSA_OVERRIDE_GFX_VERSION=10.3.0` kell (az image-ben alapból be van állítva).

### 1. Előfeltétel: a branch az origin-en
A branchet neked kell pusholnod az `origin`-re (nekem csak kifejezett kérésre szabad). Az image a forkból klónozza a `dual-buffer` branchet (`https://github.com/csaki89/llama.cpp.git`). Ha a fork privát: `export GH_TOKEN=...` a build előtt (BuildKit secret, nem kerül az image-be).

### 2. Build
```
git clone -b dual-buffer https://github.com/csaki89/llama.cpp.git && cd llama.cpp   # vagy: git pull
tools/dualbuf/build-nas.sh                 # kimenet: llama-dualbuf:rocm
```
- Alap: `rocm/dev-ubuntu-24.04:7.0-complete` (`ROCM_TAG=...` felülírja, ha más 7.x tag kell), `GPU_TARGET=gfx1030`, `BRANCH=dual-buffer`. A tag létezését a pull-nál ellenőrizni kell (nem tudtam letölteni).
- A CPU-kód Broadwellre (AVX2/FMA/F16C/BMI2, AVX512 nélkül) van fordítva, `GGML_NATIVE=OFF`; az AVX2 kell a Q4_K repackhez.
- Az image tartalmazza: `llama-completion`, `llama-server`, `llama-cli`, `llama-bench`, és a `bench.sh`-t (`/opt/llama.cpp/tools/dualbuf`, a PATH-on).

### 3. Indítás
```
docker run --rm -it --device /dev/kfd --device /dev/dri/renderD129 --group-add video \
  -v /mnt/nvme/ollama/models:/models:ro -v /mnt/nvme/dualbuf:/work llama-dualbuf:rocm
```
Az image-ben: `env | grep ^LLAMA_` legyen üres (a `bench.sh` törli és jelzi őket). Gyors próba: `llama-completion --list-devices` (ROCm0 látszik) és `-m ... -ngl 99 -n 16 -p hi`.

Az Ollama GGUF-ja a `blobs/sha256-...` fájl. Megkeresése a manifestből:
```
M=/models/manifests/registry.ollama.ai/library/qwen3-vl/8b      # modell/tag szerint
BLOB=/models/blobs/$(jq -r '.layers[]|select(.mediaType|endswith("image.model"))|.digest' $M | tr : -)
```
Figyelem: az Ollama saját motoros modelljeinél (különösen a `qwen3-vl`) a GGUF tenzornevei eltérhetnek a llama.cpp-étől; ha a betöltés hibázik, a modell szokásos GGUF-ját kell használni `/mnt/nvme/dualbuf`-ból. A `qwen3-vl` multimodális: a mérés csak a szöveges részt használja (`mmproj` nélkül).

### 4. Mérés (egy futás = egy modell, a teljes módlista)
A `bench.sh` a `modes.conf` módjait méri végig: `repack`, `host`, `host_noopoff`, `repack_noopoff`, `dual`, `dual_noopoff`, `cpu_only`, `gpu_full`. Kapcsolók: `-m MODELL`, `-g NGL`, `-n GENERÁLT_TOKENEK`, `-r ISMÉTLÉS` (alap 3), `-o KIMENETI_MAPPA`. A fix beállítások: `--load-mode none -fa auto -b 512 -ub 512 --log-verbosity 4`. Az eredmény `OUT/results.tsv`, összegzés: `tools/dualbuf/summarize.sh`.

**qwen3-vl:8b** (30 / 18 / 6 GPU-réteg, 300 token):
```
for g in 30 18 6; do bench.sh -m $BLOB_QWEN3VL -g $g -n 300 -o /work/qwen3vl-ngl$g; done
```
**Qwen2.5-1.5B** (NGL 14; a GGUF-ot másold a `/mnt/nvme/dualbuf`-ba):
```
bench.sh -m /work/qwen2.5-1.5b-instruct-q4_k_m.gguf -g 14 -n 200 -o /work/qwen25-ngl14
```
**qwen3-coder:30b** (MoE):
```
bench.sh -m $BLOB_CODER30B -g <rétegek> -n 300 -o /work/coder30b -r 1
```
- A dual-buffer RAM-többlete itt kb. **9 GB** (a repackelt CPU-s rész másolata; a naplóban a `Vulkan_Host`/`ROCm_Host` sor és a `dual-buffer: N weights also stored in ... (X MiB)` mutatja). Ellenőrizd a szabad RAM-ot a futás előtt; szükség esetén `--dual-buffer-limit MiB` / `--dual-buffer-pattern` (a `modes.conf`-ban a `dual` sorokhoz fűzhető).
- **MoE: nem tesztelt**. Az expert tenzorok (`MUL_MAT_ID`) ikerképzése és offloadja a laptopon nem volt kipróbálható. Az első futásnál nézd meg: (a) a `dual-buffer: ... weights` sor nem nulla; (b) a `dual` kimenet md5-je = `repack`-é (`dual_noopoff` esetén biztosan egyeznie kell a `repack_noopoff`-fal); (c) a prompt-sebesség a `host` szintjén van. Nagy MoE-nél a lassú mérés miatt `-r 1` elég az első körben, a végső számokhoz 3 ismétlés kell.
- A korábbi NAS-os mérések programja ismeretlen (`llama-server`?), a `bench.sh` a `llama-completion`-t használja; az összevetésnél ezt tisztázni kell. A promptok is eltérnek (a `bench.sh`: 110 / 1611 token; a korábbi: 98 token).

### 5. Várt eredmény
Generálás a `repack` szintjén, prompt a host-puffer szintjén (CLAUDE.md: +23% / +59% / +64% a repackhez képest 30 / 18 / 6 rétegnél). Ha a prompt a host szintjén van, de a generálás a lassú host-pufferes szinten marad, akkor a CPU nem az elsődleges (repack) példányt használja: ezt a naplóban a `CPU_REPACK` jelenléte és a `dual_noopoff` md5-je mutatja.

## Folyamat közben észrevettek
- A settings.json `Bash(cmd //c tools/dualbuf/build-vulkan.bat)` szabálya nem működik, mert a `cmd` perjeles útvonalat nem ért; a használható forma `cmd //c "tools\\dualbuf\\build-vulkan.bat"`. Ez javítva (a szabály most ebben az alakban van a settings.json-ban).
- Az éjszaka során egyetlen parancs sem várt engedélyre. A munkamenet engedélyezési módja megengedőbb lehetett, mint a settings.json; a használt, de az allow listán nem szereplő parancsok: `sed`, `awk`, `grep`, `cat`, `cmp`, `md5sum`, `od`, `wc`, `ls`, `ctest`, `env`, `seq`, `rm -rf bench-out`.
- Md5-összehasonlításnál a stderr-t fájlba kell irányítani; `/dev/null`-ra (Windows NUL) ANSI színkódok kerülnek a kimenetbe (lásd `test-results.md`).
- A `ctest` ebből a környezetből `BAD_COMMAND`-ot ad (msys2-s cmake); a `test-arg-parser` közvetlen futtatással átmegy.
