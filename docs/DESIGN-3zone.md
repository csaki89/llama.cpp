# Háromzónás elhelyezés, 1. építőkocka: rétegek áthelyezése futás közben

Hatókör: egy GPU-n lévő réteg súlyai futás közben, újratöltés nélkül a CPU-ra kerülhessenek és vissza. A KV-cache-hez és a zónaváltási szabályhoz (mikor mozogjon egy réteg) ez a lépés nem nyúl. Lásd ROADMAP.md.

## 1. A jelenlegi pufferkezelés
- `llama_model_loader::create_tensor()` a tenzor `buft`-jét a rétegre érvényes pufferlistából (`select_weight_buft`) vagy a `-ot` override-ból választja, majd `ctx_for_buft(buft)` egy **ggml contextet kulcsol puffertípusonként** (`ctx_key {buft, lazy}`).
- `llama_model_base::load_tensors()` minden contextre egyetlen puffert foglal (`ggml_backend_alloc_ctx_tensors_from_buft`), és a `pimpl->ctxs_bufs`-ban őrzi (context + pufferek). Egy GPU-n minden GPU-s réteg **ugyanabban** a pufferben van, ezért egy réteg VRAM-ja önállóan nem szabadítható fel.
- A gráfépítők a `model.layers[il].wq` stb. `ggml_tensor *` mutatókat használják. A mutatók a modell élettartama alatt állandók.
- A `ggml_backend_sched` a súly `tensor->buffer`-e alapján rendeli a csomópontot backendhez (`ggml_backend_sched_backend_from_buffer`), **minden gráfhasításnál** újra (`ggml_backend_sched_split_graph`). A scheduler nem cache-eli a súly elhelyezését.
- `llama_context::process_ubatch()` a gráfot újrahasznosítja, ha a paraméterei egyeznek (`gf_res_prev_active`, `can_reuse`); `sched_reserve()` a legrosszabb esetű gráfra számolja a compute-puffereket.
- Dual-buffer: a CPU-s repackelt súlyokhoz a loader host-példányt tart (`dual_pairs`), a context a schedulerbe regisztrálja (`ggml_backend_sched_set_weight_alt`).

## 2. Javasolt megoldás
Alapelv: a **modell tenzorobjektuma nem változik** (a gráfépítők mutatói érvényesek maradnak), csak az, hogy melyik pufferben/címen van az adata. Minden rugalmas sávba tartozó GPU-s súlyhoz a loader egy **CPU-s ikertenzort** hoz létre (repackelt, `select_weight_buft(..., buft_list_cpu)`), és a két tenzor tárolási mezőit (`buffer`, `data`, `extra`) cserélgeti.

### 2.1 Betöltés
- Új opció: `--flex-layers N` (`llama_model_params::flex_layers`, alapból 0 = ki). A rugalmas sáv a GPU-n lévő ismétlődő rétegek közül az utolsó N: `il` in `[max(n_layer - N, i_gpu_start), n_layer)`.
- A sáv rétegeinek GPU-s (nem host, nem split) tenzorai **rétegenként külön contextbe, így külön pufferbe** kerülnek: a `ctx_key` kap egy `flex` (rétegszám, -1 = nincs) mezőt.
- Mindegyik ilyen tenzorhoz a loader létrehoz egy ikertenzort a CPU-s contextben (azonos név, `ggml_dup_tensor`), és bejegyzi a `flex_pairs`-ba (GPU tenzor, iker, réteg, GPU-puffertípus, fájl-index és offszet). Az ikrek a név szerinti keresésből kimaradnak (mint a dual-buffer másolatai); az adat a névvel töltődik a fájlból, a repack puffer `set_tensor`-ja átrendezi.
- Feltételek: `--load-mode none` (mmap-nál figyelmeztetéssel kikapcsol), van repack extra puffer.
- A `llama_model` eltárolja, melyik `ctxs_bufs` bejegyzés tartozik az adott réteghez.

### 2.2 Levétel (`llama_model::flex_move(il, false)`)
1. A context a hívás előtt `synchronize()`-ol (nincs futó GPU-munka).
2. Minden párra a GPU tenzor és az iker `buffer/data/extra` mezőit felcseréljük: a modell tenzora mostantól a CPU-s repackelt tartalékra mutat, az iker a VRAM-ra.
3. A réteg GPU-puffere felszabadul (`ctxs_bufs[idx].second.clear()`), az ikrek mezői nullázódnak. A VRAM ténylegesen visszajön, mert a puffer a réteg sajátja.

### 2.3 Visszatétel (`flex_move(il, true)`)
1. Új GPU-puffer a réteg ikertenzoraira (`ggml_tallocr` a puffertípuson, a méret az `alloc_size` összege + igazítás).
2. Az adat **a modellfájlból** töltődik vissza (a fájl újranyitása, offszet/hossz a párban), és `ggml_backend_tensor_set`-tel kerül a VRAM-ba.
3. A mezők visszacserélődnek: a modell tenzora újra a VRAM-ra mutat, az iker a repackelt tartalékra.

### 2.4 A scheduler és a gráf
A hasítás minden gráfépítésnél a súly aktuális `buffer`-éből jön, ezért elég a gráf újraépítését kikényszeríteni (`gf_res_prev_active = nullptr`): ez a levétel/visszatétel után a context dolga. A compute-puffereket a `ggml_gallocr` szükség esetén újraméretezi. Ha a mérés mást mutat (pl. a CPU compute-puffer túl kicsi, vagy a visszatétel utáni gráf hibás), a `sched_need_reserve = true` (teljes újrafoglalás) a tartalék megoldás; ennek az eredménye a test-results.md-ben lesz.

### 2.5 Kiváltás teszteléshez
Hibakereső környezeti változó: `LLAMA_FLEX_TEST="<token>:<take|put>:<réteg>,..."`. A `llama_context::decode()` elején, ha az eddig feldolgozott tokenek száma elérte a küszöböt, végrehajtja a műveletet. Indok: nem kell szerver, nincs új API, egy helyen van egy kis blokk, `llama-completion`-nel és `llama-server`-rel is használható; a szerver hibakereső végpontja új felületet adna, és a tesztekhez nem kell.

### 2.6 Dual-bufferrel együtt
A rugalmas sáv rétegei **kimaradnak a dual-buffer ikrekből** (azok eleve GPU-s tenzorok, nincs mit duplikálni). Levétel után a réteg a repackelt tartalékkal számol a CPU-n; az op offload erre a rétegre nem megy (a repack puffer nem host). A többi, CPU-n maradó réteg dual-buffer viselkedése változatlan. (Levett rétegek host-példánya mint `weight_alt` későbbi lépés lehet: a context-eknek futás közben be kellene jegyezni a schedulerbe.)

## 3. Alternatívák (elvetve)
- **Sched-szintű csere (`weight_alt` mintára):** a modell tenzora marad a GPU-n, a scheduler egy térképen át cserélné az ikerre. Minden contexttől regisztrációt igényel és a CPU-s számítás is a "GPU-s" tenzor köré épülne; a mező-csere egyszerűbb, mert a gráf és a scheduler változatlan marad.
- **Új `ggml_tensor` mutatók a gráfépítőkben:** minden modellarchitektúrát érintene.
- **A VRAM-példány megtartása RAM-ban (eredeti formában) a visszatételhez:** gyorsabb visszatétel, de RAM-többlet rétegenként (dual-buffer-szerű). A fájlból töltés RAM-ot nem fogyaszt, és a lemezes page cache általában melegen tartja; a visszatétel költségét megmérjük (ha a mérés túl lassú, a RAM-példány opcióként jöhet).
- **Szerver hibakereső végpont:** lásd 2.5.
- **Közös puffer, részleges felszabadítás:** a puffer nem darabolható (a ROCm VMM-teszt is mutatta: a darab felszabadítása nem adja vissza a VRAM-ot, lásd ROADMAP.md), ezért rétegenkénti puffer kell.

## 4. Kockázatok
- **Több context egy modellen:** a mező-csere a modellt módosítja, így a többi contextben lévő gráf elavulhat. Ebben a lépésben csak egy context támogatott (dokumentált korlát).
- **Futó munka:** a csere előtt kötelező a szinkronizálás; a `decode()` eleje erre alkalmas.
- **Backend-gráfok (CUDA graph stb.):** a címváltozást a backendek a gráf újraépítésénél kezelik; ROCm/CUDA-n külön ellenőrizendő (a laptopon Vulkan van).
- **Többeszközös gép, split-mode row:** a split puffer tenzoraira nem alkalmazzuk (kimarad a sávból).
- **Nézetek:** ha egy réteg tenzorát nézetként használja a gráf, a nézet `data` mutatója a gráfépítéskor jön a szülőből, ezért újraépítés után helyes.
- **Teljesítmény:** a visszatétel egy rétegnyi fájlolvasás + feltöltés; mérjük.
- **Rebase:** `llama-model-loader.cpp/.h`, `llama-model.cpp/.h`, `llama-context.cpp`, `llama.h`, `common/arg.cpp`, `common/common.cpp/.h`.
