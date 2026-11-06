IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# CLAUDE.md (dual-buffer fork)

## Szabályok
- Soha ne nyiss pull requestet és ne tölts fel semmit a ggml-org/llama.cpp repóba. Az upstream remote csak frissítések lehúzására való (a push URL `DISABLED`).
- Feltöltés csak a saját forkba (origin), és előtte mindig kérdezz rá.
- A gh alapértelmezett repója a saját fork (csaki89/llama.cpp).
- A munka a `dual-buffer` branchben folyik, a master tiszta marad.

## Viszony az AGENTS.md-hez
Ez egy privát fork, nem a ggml-org/llama.cpp-be szánt hozzájárulás. Az AGENTS.md szerint a privát forkok mentesek ("Private forks are exempt"), ezért ebben a repóban:
- A dual-buffer branchen a commitok előre jóvá vannak hagyva, nem kell egyenként rákérdezni. A commit üzenetet te írod: rövid, és tartalmazza: `Assisted-by: Claude`. Upstream PR esetén az üzeneteket a felhasználó írja újra.
- A `docs/DESIGN.md`-ben rögzített terv megvalósításához nem kell külön megerősítés, akkor sem, ha a változás nagy vagy új mintát hoz be.
- Az önálló (felügyelet nélküli) munka ebben a forkban engedélyezett, mert nem az upstreambe történik hozzájárulás.
- Továbbra is tilos: git push az upstreambe, PR nyitása, issue vagy komment írása a ggml-org/llama.cpp-ben. A saját forkba (origin) pusholni csak kifejezett kérésre lehet.

Az AGENTS.md kódolási előírásai (ASCII-karakterek, rövid kommentek, a meglévő infrastruktúra újrahasznosítása, minimális és nem invazív változtatás, a `skills/code-review/SKILL.md` elolvasása kódírás előtt) érvényesek.

## Cél
Dual-buffer a llama.cpp-ben hibrid (CPU+GPU split) futtatáshoz. Ma minden súlytenzornak egy helye és formája van, ezért a CPU-n maradó súlyoknál választani kell:
- **CPU_REPACK** (AVX2-re átrendezve): gyors CPU-s generálás, de az op offload nem tudja a GPU-ra küldeni, így a prompt lassú.
- **Host-puffer** (ROCm_Host / Vulkan_Host, eredeti formátum, pinned memória): az op offload a GPU-ra küldi, gyors prompt, de a CPU nem repackelt súlyokkal számol, lassú generálás.

A dual-buffer mindkét példányt megtartja: a CPU a repackelt példánnyal számol, az op offload az eredeti formájút küldi át.

## Mért tények
NAS: 2× Xeon E5-2673 v4, 128 GB DDR4, AMD RX 6700 10 GB / gfx1031, ROCm, qwen3-vl:8b Q4_K_M, 98 tokenes prompt, 300 token generálás.

| GPU-rétegek | Repack prompt / gen (tok/s) | Host-puffer prompt / gen | Host-puffer, --no-op-offload prompt / gen |
|---|---|---|---|
| 30 | 282–290 / 28,7–29,5 | 356 / 20,2 | 220 / 21,2 |
| 18 | 149–170 / 18,8–20,0 | 238 / 10,2 | 114 / 10,9 |
| 6 | 110–118 / 14,0–14,5 | 189 / 6,9 | 78 / 7,3 |

Következtetés: a host-puffer prompt-előnye teljes egészében az op offloadból jön. Várt eredmény dual-bufferrel: generálás a repack szintjén, prompt a host-puffer szintjén (+23% / +59% / +64%).

Fork nélkül nem kombinálható: az `-ot` célként csak CPU és ROCm0 fogadható el, és a CPU cél is a host-pufferbe kerül (a pufferlista sorrendje: eszközök, device host buffer, extra bufferek (CPU_REPACK), CPU).

## Kódpontok
- `llama-model.cpp`: `make_cpu_buft_list()` (pufferprioritás), `get_layer_buft_list()`, `select_weight_buft()` (az első támogató puffert választja)
- ggml-cpu: a repack extra buffer
- ggml-backend scheduler: az op offload döntése
- Kapcsolódó upstream issue: ggml-org/llama.cpp#31538 (host-puffer prioritás)

## Tervezési elvek
- Lehetőleg ne legyen benne backend-specifikus kód: ugyanaz a logika fusson Vulkan és ROCm alatt.
- A duplikálás legyen kikapcsolható, és később korlátozható tenzorcsoportra vagy RAM-keretre: a CPU-s rész kétszer tárolva nagy modelleknél (gpt-oss:120b, ~57 GB CPU-s rész) nem fér el a 128 GB-ban.
- Későbbi lépés: háromzónás kiosztás (mindig GPU / rugalmas sáv repackelt tartalékkal / mindig CPU).

## Munkamenet
- **Laptop** (Windows, Vulkan, iGPU): fejlesztés, fordítás, helyességi tesztek kis modellel (azonos kimenet temperature 0 mellett, mindkét puffer a naplóban, a prompt a host-példányt küldi át). Sebességmérésre nem alkalmas, mert az iGPU ugyanazt a RAM-ot használja.
- **NAS** (TrueNAS, Docker, ROCm): a branch ROCm-re fordítva, sebességmérés ugyanazzal a sorozattal (30/18/6 réteg). Referencia-beállítások: `--load-mode none --flash-attn auto -b 512 -ub 512 --log-verbosity 4`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Figyelem: a `LLAMA_ARG_NO_HOST` környezeti változó felülírja a viselkedést, méréskor ellenőrizni kell.

## Laptop build (Windows, natív)
- Eszközök: VS 2022 Community (MSVC 14.44, Windows SDK 26100), a VS-be épített CMake és Ninja, Vulkan SDK `C:\VulkanSDK\1.4.363.1`. Az msys2 cmake/git nem a natív toolchain része.
- Fordítás: vcvars64.bat környezetben, `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DLLAMA_CURL=OFF`, majd `cmake --build build -j 6`.
- Teszt modell: `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (nincs verziókezelve).
