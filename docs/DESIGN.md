# Dual-buffer terv

Állapot: 1. fázis (kódfelderítés), a kód ennél a commitnál még módosítatlan. Az `src/`, `ggml/` hivatkozások a `master` `e14a77971` állapotára vonatkoznak.

## 1. Hogyan működik ma

### 1.1 Pufferválasztás a betöltésnél
- `src/llama-model.cpp:1228` `make_cpu_buft_list(devices, use_extra_bufts, no_host)` építi a CPU-n maradó súlyok prioritáslistáját, ebben a sorrendben:
  1. ACCEL eszközök pufferei (ha vannak),
  2. az első eszköz host-puffere (`ggml_backend_dev_host_buffer_type`, Vulkan_Host / ROCm_Host / CUDA_Host), ha `!no_host`,
  3. a CPU backend extra pufferei (`use_extra_bufts`): ezek közül jön létre a `CPU_REPACK`,
  4. a sima CPU puffer.
- `src/llama-model.cpp:1718` ezt a listát minden GPU listájának végére is hozzáfűzi tartalékként, a CPU-n lévő rétegek (`dev_input`, a nem offloadolt rétegek) pedig közvetlenül ezt használják (`get_layer_buft_list`, `:1766`).
- **Eltérés a CLAUDE.md-hez képest:** a `select_weight_buft()` nem a `llama-model.cpp`-ben van, hanem a `src/llama-model-loader.cpp:1080`-ban, és a tenzorlétrehozás (`llama_model_loader::create_tensor`, `:1129`) hívja. A függvény a lista első olyan pufferét adja vissza, amelyre `weight_buft_supported()` igaz (egy teszt-op `supports_op` lekérdezése egy 0 méretű ideiglenes pufferrel).
- Mivel a host-puffer a listában a repack előtt van és (a legtöbb opra) támogat, ezért alapból az nyer. `--no-host`-tal kimarad, így a repack nyer a repackelhető tenzorokra (Q4_K stb.), a többi (pl. token embedding) a sima `CPU` pufferbe kerül.
- `:1282` mmap esetén a host-puffer választását felülírja a sima CPU puffer ("avoid using a host buffer when using mmap").
- Egy tenzor egy `ggml_context`-be kerül, pufferenként egy contextben (`ctx_map`, `:1135`). A pufferek lefoglalása és a `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` jelölés a `llama_model::load_tensors()`-ban történik (`llama-model.cpp:1990-2080`), az adatok betöltése a `load_all_data()`-val, név szerint. A repack puffer `set_tensor`-ja végzi az átrendezést.
- Már van minta kettős tenzorra: `TENSOR_DUPLICATED` (`llama-model-loader.cpp:1384-1399`), azonos nevű tenzor más contextben, a `n_created` nem nő, a `size_data` igen.

### 1.2 A repack extra puffer
- `ggml/src/ggml-cpu/repack.cpp:5253`: a repack puffertípus `is_host = nullptr`, vagyis `ggml_backend_buft_is_host()` hamis rá. A CPU backend `supports_buft` (`ggml-cpu.cpp:490`) elfogadja ("host vagy extra puffer"), a `supports_op` (`ggml-cpu.cpp:424-440`) az extra pufferhez tartozó `supports_op`-ra delegál.
- A repackelt adat a pufferben átrendezett, nem az eredeti ggml formátum, ezért semmi más backend nem tudja olvasni.

### 1.3 A scheduler op-offload döntése
- `ggml/src/ggml-backend.cpp:1329` `ggml_backend_sched_backend_id_from_cur()` (1. pass): a csomópontot ahhoz a backendhez rendeli, ahol a súlya van (`ggml_backend_sched_backend_from_buffer`: az első backend, amelyre `supports_buft && supports_op`).
- `:1375-1388`: ha a súly a CPU backenden van (utolsó backend), és `sched->op_offload`, és **`ggml_backend_buffer_is_host(src->buffer)`**, akkor a magasabb prioritású backendek közül az első, amelyre `supports_op && offload_op` (batch >= küszöb, Vulkan: `op_offload_min_batch_size`), kapja a csomópontot.
- Ezután az 5. pass (`:1773-1805`) a súlyt, mint a split bemenetét, átmásolja a GPU-ra, ha a GPU backend nem támogatja a puffer típusát (host-puffer: igen, ilyenkor a másolás gyors, mert rögzített memória).
- **Miért bukik el a repackelt súlynál:** a `ggml_backend_buffer_is_host(CPU_REPACK)` hamis (1.2), ezért a `:1378` feltétel sosem teljesül, a csomópont a CPU-n marad. Ha mégis átmásolnánk, a GPU az átrendezett adatot kapná.

### 1.4 Hol alakul ki a végeredmény
`llama_context::sched_new()` (`src/llama-context.cpp:783`) hozza létre a schedulert. A gráf minden `ggml_backend_sched_split_graph` hívásnál újraosztódik, a 5. pass már most is átírja a csomópontok `src[]` mutatóit (`:1803`), tehát a gráf csomópontjainak módosítása bevett.

## 2. Javasolt megoldás

Alapelv: a modell két tenzort tart minden repackelt CPU-s súlyhoz: az elsődlegest (`CPU_REPACK`, ezt használja a gráf és a CPU), és egy ikertenzort (eredeti formátum, a GPU backend host-pufferében). A scheduler, ha egy csomópontot a GPU-ra offloadol, az elsődleges súlyt az ikerre cseréli.

### 2.1 Betöltés (`llama-model-loader.cpp`, `llama-model.cpp`)
1. Új `llama_model_params::dual_buffer` (alapból `false`), `--dual-buffer` / `LLAMA_ARG_DUAL_BUFFER`.
2. `load_tensors()`: ha be van kapcsolva és van használható host-puffertípus (az első eszköz `ggml_backend_dev_host_buffer_type`-ja, ugyanaz a logika, mint `make_cpu_buft_list()`-ben), és nincs mmap, és `use_extra_bufts`:
   - a CPU lista **host-puffer nélkül** épül (mintha `--no-host`), így az elsődleges tenzor a `CPU_REPACK`-be kerül;
   - a loadernek átadjuk a host-puffertípust (`ml.dual_host_buft`).
3. `llama_model_loader::create_tensor()`: ha az elsődleges tenzor puffere nem host puffer, és az eszköze CPU/ACCEL típusú (vagyis extra puffer, pl. `CPU_REPACK`), és `weight_buft_supported(…, dual_host_buft, …)` igaz, akkor ugyanabból a `t_meta`-ból egy második tenzor jön létre a host-puffer contextjében (azonos név, `ggml_dup_tensor`), és a pár bekerül `ml.dual_pairs`-ba. A `size_data` nő (progress), a `n_created` nem. Backend-független: csak puffertípus-tulajdonságokra támaszkodik, nem Vulkan/ROCm-specifikus.
4. Az ikertenzor `load_all_data()`-val, név szerint töltődik, a repack puffer nélkül (a host-puffer `set_tensor`-ja sima másolás).
5. Az ikertenzorok **nem** kerülnek be a `tensors_by_name`-be (`llama-model.cpp:1978`), mert a `get_tensor(name)` az első egyezést adja, és a LoRA, az `n_elements` és más név szerinti keresők az elsődlegest várják.
6. A párok a `llama_model`-ben tárolódnak (`dual_weights`).
7. Napló: egy sor `dual-buffer: N weights duplicated, X MiB in <puffernév>`, a pufferméretek (`CPU_REPACK`, `Vulkan_Host`) a meglévő "model buffer size" sorokban látszanak.

### 2.2 Scheduler (`ggml-backend.h/.cpp`)
- Új API: `void ggml_backend_sched_set_weight_alt(ggml_backend_sched_t sched, struct ggml_tensor * weight, struct ggml_tensor * alt);` (a scheduler egy `unordered_map`-et tart, a `sched_free` felszabadítja).
- `ggml_backend_sched_backend_id_from_cur()`: a `:1378` feltétel kiegészül: `(ggml_backend_buffer_is_host(src->buffer) || alt != NULL)`. Ha van iker, a `src[i]`-t ideiglenesen az ikerre cseréljük, és a `supports_op && offload_op` ellenőrzést így végezzük; ha egy backend elfogadja, a csere megmarad, és az iker backend-azonosítója a CPU lesz (`tensor_backend_id(alt) = src_backend_id`), így az 5. pass a meglévő host-puffer útvonalon másolja át. Ha egyik backend sem fogadja el, visszaállítjuk az eredeti `src[i]`-t.
- A decode (`n_tokens` kicsi) `offload_op` hamis, ezért a CPU az elsődleges (repack) tenzort használja. A prompt (nagy batch) a GPU-ra megy, az iker host-példánnyal.
- `llama_context::sched_new()`: a sched létrehozása után a `model.dual_weights` párjait regisztráljuk.

### 2.3 Memória és `--fit`
- A RAM-többlet pontosan a repackelt CPU-s súlyok mérete (a laptopos modellnél a `CPU_REPACK` 308,18 MiB).
- A `--fit` a modellt no_alloc módban tölti be ugyanazzal a kóddal (`ml.no_alloc`), tehát az ikertenzorok a host-puffer méretébe belekerülnek. A host-puffer nem eszközmemória, ezért a `--fit` eszközmemória-számítását nem érinti. A host RAM többletét a `--fit` nem ellenőrzi, ez korlát, a SUMMARY.md-ben rögzítendő (a 4. fázis RAM-kerete ezt oldaná meg).

## 3. Alternatívák (elvetve)
- **A: Külön tenzor a gráfépítésnél** (modellfüggetlen `build_lora_mm` helyett kétféle súlyt választani): a döntés a schedulerben dől el, nem a gráfépítéskor, és minden modell grafikonját érintené.
- **B: Az iker mutatója a `ggml_tensor`-ban:** a `ggml_tensor` szerkezetének bővítése minden backendet és ABI-t érint. A scheduler-oldali map helyette lokális.
- **C: A repack puffer host-nak jelölése, a másolásnál visszaalakítás:** minden kiértékeléskor CPU-n kellene visszarendezni, ez lassabb, mint a host-példány.
- **D: Dupla pufferlista tenzoronként a loaderben, scheduler nélkül:** a scheduler akkor sem tudná, hogy melyik példányt küldje át.

## 4. Kockázatok
- **Hash-méret:** a scheduler hash-halmazának mérete a gráfmérethez igazodik (`sched_new`, "FIXME: needs size*2"). Az ikertenzorok új bejegyzések, legfeljebb a súlyok számával. A tipikus `graph_max_nodes` bőven ad tartalékot, de óriási modellnél (sok súly) ellenőrizendő.
- **Nézetek (view):** ha egy csomópont a súly egy nézetét kapja `src`-ként (pl. szeletelt expert), az iker-keresés nem talál, és a viselkedés a mai marad (nincs offload). A MUL_MAT_ID közvetlenül a teljes tenzort kapja, ezt a 3. fázisban dense modellel ellenőrizzük, MoE-vel nem.
- **mmap:** host-puffer és mmap nem kombinálható (`llama-model-loader.cpp:1282`), ezért a dual-buffer `--load-mode none` nélkül kikapcsol, figyelmeztetéssel.
- **Gráf-újrahasznosítás:** a `src` csere a `split_graph` részeként történik, ugyanúgy, mint az 5. pass másolatai, ezért a gráf-újrahasználat nem ront rajta, de a 3. fázis teszteli.
- **Rebase:** az érintett upstream fájlok: `ggml-backend.cpp/.h`, `llama-model-loader.cpp/.h`, `llama-model.cpp/.h`, `llama-context.cpp`, `llama.h`, `common/arg.cpp`, `common/common.cpp/.h`. A módosítások kis, jól körülhatárolt blokkok.

## 5. Upstream keresés (`gh search`)
- Keresések: "dual buffer", "op offload repack", "CPU_REPACK offload", "extra buffer op_offload", "host buffer prompt processing repack", "no-host" (issue-k és PR-ok).
- Eredmény: nincs olyan issue vagy PR, amely ezt a megoldást vagy egy hasonlót tartalmazna. Az egyetlen kapcsolódó a #31538 (a címe a keresés szerint csak "Misc. bug:", nyitott, `bug-unconfirmed`). A tartalmát nem tudtam elolvasni, mert a `gh issue` és a `gh api` ebben a projektben tiltott, így a CLAUDE.md-ben leírt "host-puffer prioritás" témát nem ellenőriztem.
