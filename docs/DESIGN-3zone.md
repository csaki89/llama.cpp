# Háromzónás elhelyezés, 1. építőkocka: rétegek áthelyezése futás közben

## A zónák értelmezése (érvényes)
1. **zóna, mindig GPU:** ami a maximális kontextus KV-cache-e mellett is elfér; csak a VRAM-ban.
2. **zóna, rugalmas sáv:** a RAM-ban mindig két formában van meg: GPU-formájú (eredeti) példány a host-pufferben és CPU-ra repackelt tartalék; amikor a réteg a GPU-n van, a VRAM-ban is.
3. **zóna, mindig CPU:** a RAM-ban CPU-ra repackelve, és ha a dual-buffer be van kapcsolva, mellette a GPU-formájú példány a prompt op offloadhoz.

A levett 2. zónás réteg ugyanúgy viselkedjen, mint egy eleve CPU-n lévő dual-bufferes réteg: a súlyai és a K/V-je a CPU-n, a prompt op offloadja a host-példányból.

**A megvalósítás állapota ehhez képest** (`--flex-layers`, lásd 2. és 5.): kész. A rugalmas sáv a GPU-s rétegek első N rétege, a repackelt tartalék, a host-példány (op offload levett állapotban, a visszatétel forrása) és a K/V követése megvan; egy végig levett réteg kimenete, VRAM-ja és gráfhasításai bájtra/számra azonosak egy eleve CPU-s dual-bufferes rétegével (`-ngl 13 --dual-buffer`, test-results.md). Nyitott: a zónaváltási szabály, a dinamikus KV, a foglalások költsége (Következő lépések).

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
- Új opció: `--flex-layers N` (`llama_model_params::flex_layers`, alapból 0 = ki). A rugalmas sáv a GPU-n lévő ismétlődő rétegek közül az **első N, közvetlenül a CPU-s rétegek mellett**: `il` in `[i_gpu_start, min(i_gpu_start + N, n_layer))`. A levétel a CPU-GPU határ felől halad (`llama_model::flex_next`), a visszatétel fordítva, így egy levett réteg csak a határt tolja el (DECISIONS.md 15).
- A sáv rétegeinek GPU-s (nem host, nem split) tenzorai **rétegenként külön contextbe, így külön pufferbe** kerülnek: a `ctx_key` kap egy `flex` (rétegszám, -1 = nincs) mezőt.
- Mindegyik ilyen tenzorhoz a loader létrehoz egy ikertenzort a CPU-s contextben (azonos név, `ggml_dup_tensor`; a puffertípus csak a CPU-eszköz sajátjai közül: repack, ha lehet, különben CPU), és ha az iker repackelt, egy **eredeti formájú host-példányt** a GPU-eszköz host-pufferében (rögzített memória, 2. zóna). A pár a `flex_pairs`-ba kerül (GPU tenzor, iker, host-példány, réteg, GPU-puffertípus, fájl-index és offszet). Az ikrek és a host-példányok a név szerinti keresésből kimaradnak; az adat a névvel töltődik a fájlból, a repack puffer `set_tensor`-ja átrendezi. A nem repackelhető súlyok (normák, biasok) ikre eleve host-olvasható, ezért azokhoz nincs második példány (DECISIONS.md 17).
- RAM-védelem: a sáv `2 x` méretével számol; ha nem fér el, a host-példányok elmaradnak (tartalék: visszatétel a fájlból), ha az ikrek sem, a sáv kikapcsol (DECISIONS.md 18).
- Feltételek: `--load-mode none` (mmap-nál figyelmeztetéssel kikapcsol).
- A `llama_model` eltárolja, melyik `ctxs_bufs` bejegyzés tartozik az adott réteghez. A `dev_layer(il)` levett rétegnél a CPU-t adja, ahogy egy CPU-s rétegnél (DECISIONS.md 16).

### 2.2 Levétel (`llama_model::flex_move(il, false)`)
1. A context a hívás előtt `synchronize()`-ol (nincs futó GPU-munka).
2. Minden párra a GPU tenzor és az iker `buffer/data/extra` mezőit felcseréljük: a modell tenzora mostantól a CPU-s repackelt tartalékra mutat, az iker a VRAM-ra.
3. A réteg GPU-puffere felszabadul (`ctxs_bufs[idx].second.clear()`), az ikrek mezői nullázódnak. A VRAM ténylegesen visszajön, mert a puffer a réteg sajátja.

### 2.3 Visszatétel (`flex_move(il, true)`)
1. Új GPU-puffer a réteg ikertenzoraira (`ggml_tallocr` a puffertípuson, a méret az `alloc_size` összege + igazítás).
2. Az adat forrása: a **host-példány** (repackelt ikrű súlyok), a host-olvasható CPU-iker (nem repackelt súlyok), tartalékként a **modellfájl** (ha a host-példány nem foglalható le, vagy `LLAMA_FLEX_TEST_FILE=1`). A másolás szinkron `ggml_backend_tensor_set`: rögzített forrásból a backend DMA-val ír, UMA-n közvetlenül (DECISIONS.md 19).
3. A mezők visszacserélődnek: a modell tenzora újra a VRAM-ra mutat, az iker a repackelt tartalékra; a context kiveszi a host-példányokat a schedulerből.

### 2.4 A scheduler és a gráf
A hasítás minden gráfépítésnél a súly aktuális `buffer`-éből jön, ezért elég a gráf újraépítését kikényszeríteni (`gf_res_prev_active = nullptr`): ez a levétel/visszatétel után a context dolga. A compute-puffereket a `ggml_gallocr` szükség esetén újraméretezi. Ha a mérés mást mutat (pl. a CPU compute-puffer túl kicsi, vagy a visszatétel utáni gráf hibás), a `sched_need_reserve = true` (teljes újrafoglalás) a tartalék megoldás; ennek az eredménye a test-results.md-ben lesz.

### 2.5 Kiváltás teszteléshez
Hibakereső környezeti változó: `LLAMA_FLEX_TEST="<token>:<take|put>:<réteg>,..."`. A `llama_context::decode()` elején, ha az eddig feldolgozott tokenek száma elérte a küszöböt, végrehajtja a műveletet. Indok: nem kell szerver, nincs új API, egy helyen van egy kis blokk, `llama-completion`-nel és `llama-server`-rel is használható; a szerver hibakereső végpontja új felületet adna, és a tesztekhez nem kell.

### 2.6 Dual-bufferrel együtt
A rugalmas sáv rétegei a dual-buffer ikrekből kimaradnak (GPU-s tenzorok), helyettük a saját host-példányuk van (2.1). **Levett állapotban a context a host-példányokat bejegyzi a schedulerbe** (`ggml_backend_sched_set_weight_alt`), visszatételkor `NULL`-ra állítja, és az újralétrehozott schedulerbe (`sched_new`) a pillanatnyi állapot szerint jegyzi be. Így a levett réteg generáláskor a repackelt tartalékkal számol a CPU-n, a prompt op offloaddal a host-példányt küldi a GPU-ra: pontosan úgy, mint egy eleve CPU-s dual-bufferes réteg (mérés: `-ngl 13 --dual-buffer`-rel bájtra azonos kimenet). Ez a `--dual-buffer` nélkül is így működik (a host-példány a sáv része). A többi réteg dual-buffer viselkedése változatlan. `LLAMA_FLEX_TEST_HOST=0` kikapcsolja a bejegyzést (összehasonlításhoz).

### 2.7 A réteg KV-cache-e követi a réteget
Ok: ha a K/V a GPU-n marad, a levett rétegben a mátrixszorzás a CPU-n, a figyelemszámítás (amely a K/V pufferéhez rendelődik) a GPU-n fut, ez tokenenként extra eszközváltásokat (gráfhasításokat) okoz, és a levétel kevesebb VRAM-ot szabadít fel. A llama.cpp alapból is annak az eszköznek a memóriájába teszi a réteg K/V-jét, ahol a réteg fut (`llama_kv_cache` konstruktor, `model.dev_layer(il)`), a levett réteg így ugyanúgy viselkedik, mint egy eleve CPU-n lévő.
- **Betöltés:** a `llama_kv_cache` a flex rétegek (`model.flex_has_layer(il)`, csak `offload` mellett) K/V tenzorait rétegenként saját contextbe és pufferbe teszi (`flex_kv_layers`: a puffer indexe a `ctxs_bufs`-ban, a GPU-puffertípus, az aktuális hely). A többi réteg K/V-je a megszokott, puffertípusonként közös pufferben marad.
- **Mozgatás (`llama_kv_cache::flex_move(il, to_gpu, backend)`):** új puffer a céleszközön (CPU: a GPU-eszköz host-puffere, rögzített memória, amit a CPU-backend host-pufferként használ; tartalék a CPU-puffer; GPU: az eredeti), a K és V újrafoglalása benne (`ggml_tallocr`), **csak a használt cellák** átmásolása (`[0, used_max_p1)` soronként, transzponált V-nél a teljes tenzor), GPU -> CPU irányban a backend aszinkron olvasásával a rögzített memóriába (DMA) + szinkronizálás, CPU -> GPU irányban szinkron `set`-tel; a stream-nézetek (`k_stream`/`v_stream`) címének frissítése, végül a régi puffer felszabadítása. A cellák (pozíciók, sorozatok) nem változnak, a tartalom bájtra azonos, ezért nem kell újraszámolni a promptot (DECISIONS.md 19-20).
- **Sorrend:** a context a súlyok mozgatása után mozgatja a K/V-t, ugyanabban a lépésben, szinkronizálás után, majd a következő gráf újraépül.
- **Hatókör:** csak a sima `llama_kv_cache` (dinamikus cast a contextben); más memóriatípusnál (iSWA, hibrid, rekurrens) a K/V helyben marad, figyelmeztetéssel. A KV mérete nem változik (a dinamikus KV-méretezés nem része ennek a lépésnek).
- **Összehasonlításhoz:** `LLAMA_FLEX_TEST_KV=0` helyben hagyja a K/V-t (az előző viselkedés).

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
- **ROCm/HIP: a HIP-gráfok állapota.** A HIP backend (a CUDA-kód HIP-fordítása) a decode-gráfokat HIP-gráfként rögzítheti és újrajátszhatja; a rögzített gráf a súlyok és a K/V címét tartalmazza. A mozgatás után ellenőrizni kell a NAS-on, hogy a backend felismeri-e a címváltozást és újrarögzít-e (a csomópontok `src` adatcímeinek összevetése), és hogy egy felszabadított pufferre mutató rögzített gráf nem fut-e le (`GGML_CUDA_DISABLE_GRAPHS=1`-gyel összehasonlítva).
- **A K/V másolása:** csak a használt cellák másolódnak, de a puffer foglalása és törlése a teljes lefoglalt K/V-re vonatkozik (a kontextusmérettel nő), és a másolás alatt rövid ideig mindkét példány létezik (csúcs-memória a céleszközön + a forráson). Transzponált V (flash attention nélkül) teljes másolást jelent.
- **Aszinkron olvasás:** a GPU -> CPU másolás a backend aszinkron `get`-jét használja egy veremben lévő tenzor-másolattal (`old`); ez a meglévő backendeknél (Vulkan, CUDA/HIP) helyes, mert a hívás csak a címet veszi át, de egy backend, amely a tenzormutatót a szinkronizálásig megtartaná, hibázna.
- **PCIe:** dedikált GPU-n a host-példány és a K/V másolása a PCIe-n megy; a NAS-on mérendő (SUMMARY.md).
- **Nem sima KV-cache:** iSWA/hibrid/rekurrens memóriánál a K/V nem mozog (2.7).
- **Többeszközös gép, split-mode row:** a split puffer tenzoraira nem alkalmazzuk (kimarad a sávból).
- **Nézetek:** ha egy réteg tenzorát nézetként használja a gráf, a nézet `data` mutatója a gráfépítéskor jön a szülőből, ezért újraépítés után helyes.
- **Teljesítmény:** a visszatétel a host-példányból egy rétegnyi feltöltés (laptopon 6-10 ms/réteg), a fájlból tartalékként lassabb (20-41 ms, meleg page cache).
- **Rebase:** `llama-model-loader.cpp/.h`, `llama-model.cpp/.h`, `llama-kv-cache.cpp/.h`, `llama-context.cpp/.h`, `llama.h`, `common/arg.cpp`, `common/common.cpp/.h`.

### Következő lépések
- ~~A levett rétegek host-példánya (op offload)~~: kész (2.1, 2.6).
- **A foglalások költsége.** A mozgatás idejét a mérés szerint a pufferfoglalás viszi (rögzített host-memória 5-18 ms, Vulkan eszközmemória 1-52 ms egy 32 MiB-os K/V-hez), nem a másolás (0,2-1,3 ms). Lehetőség: a levett rétegek host-oldali K/V-pufferének megtartása visszatétel után (RAM-költség a sáv K/V-mérete), vagy a ROADMAP VMM-terve (előre lefoglalt címtartomány).
- **A zónaváltási szabály** (mikor menjen le/vissza egy réteg) és a **dinamikus KV-méretezés** (ROADMAP.md); a mozgatást ma csak a `LLAMA_FLEX_TEST` váltja ki.
- Több context egy modellen, iSWA/hibrid memória, `llama-server` (egy kérés közbeni mozgatás).

## 5. Megvalósítás: eltérések és mért eredmény
- **Az iker puffertípusa:** a CPU-s pufferlistában alapból a GPU host-puffere áll elöl (repack előtt), ezért a `select_weight_buft(buft_list_cpu)` az ikreket a host-pufferbe tette volna, repack nélkül. Az iker ezért csak a CPU-eszköz saját puffertípusai közül választ (extra/repack, majd CPU).
- **Scheduler:** a 2.4 pont feltevése igaz: a gráf újraépítése elég, a CPU compute-puffert a `ggml_gallocr` magától átméretezi; a teljes újrafoglalás (`LLAMA_FLEX_TEST_RESERVE=1`) ugyanazt a kimenetet adja, ezért nem kell.
- **Fájlok:** a loader `files` listája a betöltés végén a modellhez kerül (a loadert a betöltés után semmi nem használja), a visszatétel ezekből olvas.
- **Mért értékek** (laptop, részletek: test-results.md): levétel 0,6-3,2 ms/réteg, visszatétel 20-67 ms/réteg (~27 MiB), a modell VRAM-foglalása a levett rétegek méretével csökken (526 -> 472 MiB két réteggel).
- **K/V mozgatása** (2.7, részletek: test-results.md): két levett réteggel a felszabaduló VRAM 54 MiB-ról 118 MiB-ra nő (súlyok + 2 x 32 MiB K/V), a generálási gráf hasításai 14-ről 10-re csökkennek. A K/V másolása rétegenként (32 MiB) GPU -> CPU 148-234 ms, CPU -> GPU 11-18 ms az iGPU-n. A maradék hasítások a reziduális út súly nélküli műveleteiből jönnek, amelyeket a scheduler a GPU-n hagy (alapviselkedés egy GPU-s réteg mögötti CPU-s rétegnél).

## 6. Automatikus működés (2. építőkocka)
Cél: kézi beállítás nélkül, a `--auto-zones` kapcsolóval (`LLAMA_ARG_AUTO_ZONES`, alapból ki) a program maga osztja ki a zónákat, a KV a használattal nő, és a rugalmas sáv rétegei a KV növekedésével lekerülnek, új beszélgetésnél visszakerülnek. A `--flex-layers` és a `LLAMA_FLEX_TEST` hibakereső felülírásként megmarad.

### 6.1 A sáv mérete (common, a `--fit` mérésével)
- A `common_fit_apply()` `--auto-zones` mellett **kétszer** futtatja a `--fit`-et, saját pufferekkel: (A) a beállított maximális kontextussal (`n_ctx`, illetve amit a fit választ), (B) egy kis kezdő kontextussal (`--zones-ctx-init`, alap 4096, legfeljebb az A kontextusa). Mindkettő a GPU-rétegszámot (`n_gpu_layers`) adja ugyanabból, amiből a `--fit` dolgozik (szabad VRAM, rétegenkénti súlyméret, a KV és a compute-puffer mért mérete, a `--fit-target` tartalék).
- 1. zóna = az A rétegszáma (`ngl_full`), 1. + 2. zóna = a B rétegszáma (`ngl_small`). A modell a B rétegszámával töltődik be, a sáv a különbség: új modellparaméter `flex_ngl_full` (az A eredménye); a libllama a sávot `n_gpu_layers - flex_ngl_full` rétegnek veszi (a GPU-s rétegek első rétegei a határ mellett, ahogy a `--flex-layers`). Ha az A-ban minden elfér (`ngl_full` = minden réteg), a sáv 0. A `--flex-layers N` kézzel felülírja.
- Korlát: ha a felhasználó `-ngl`-t ad, vagy a fit tensor-override-okat állít (MoE), az automatikus sáv kimarad (naplózva), a dinamikus KV megmarad.

### 6.2 Dinamikus KV (libllama, backend-független tartalék)
- Új kontextparaméter `n_ctx_init` (0 = ki, a régi viselkedés). Bekapcsolva a `llama_context` a sima `llama_kv_cache`-t `n_ctx_init` cellával hozza létre, és megadja a maximumot (`n_ctx_seq`): `llama_kv_cache::set_dynamic(max, chunk)`. Más memóriatípus (iSWA, hibrid, rekurrens, megosztott/újrahasznosított rétegek, MTP) esetén kikapcsol, naplózva. Az `n_ctx` (a kliensek felé) a maximum marad.
- **Növelés darabokban** (`llama_kv_cache::resize(n)`): a darab a ROADMAP szerint bájtban adott (alap 64 MiB az összes réteg K/V-jére), cellára kerekítve (256 többszöröse). Rétegenként: új context és puffer **ugyanazon a puffertípuson**, ahol a réteg K/V-je most van (GPU, host, CPU), törlés, a **használt sorok** (`[0, used_max_p1)` streamenként) átmásolása a régi tárolóból a backend saját útján (azonos GPU-n `cpy_tensor`, azaz eszközön belüli DMA; host <-> host `memcpy`), a nézetek újraépítése, a régi puffer felszabadítása. Rétegenként halad, így a csúcs-memória egy réteg új K/V-je (az első átméretezésnél a közös kezdőpuffer felszabadítása a végén). A cellák metaadatai megmaradnak (`llama_kv_cells::resize_keep`).
- **Csökkentés:** ugyanez kisebb mérettel, ha a használt cellák elférnek.
- Feltétel: nem transzponált V (flash attention), mert a transzponált V sorai a mérettel együtt mozognak; transzponált V-nél a dinamikus KV kikapcsol (naplózva). A figyelemszámító kernelekhez nem kell nyúlni: a gráf a K/V-t ma is `n_kv` cellás nézeteken éri el, a teljes méret csak a nézetek lépésközében szerepel.
- **VMM (CUDA/HIP):** a csatlakozási pont a `resize`: egy VMM-képes backendnél a K/V tenzorok egy előre lefoglalt címtartományban nőnének (ROCm-en 64-256 MiB-os darabokkal), másolás nélkül. Ehhez egy új, opcionális backend-API kell (növelhető puffer, `ggml_backend_reg_get_proc_address`-szel lekérdezve), a ggml-cuda/HIP VMM-kódjával. **Ebben a lépésben nem készül el**, mert a laptopon nincs CUDA/HIP, és tesztelés nélkül nem adható ki (DECISIONS.md); a tartalék (átméretezés másolással) minden backenden működik.

### 6.3 Zónaváltási szabály (libllama, `llama_context`)
- **VRAM-keret könyveléssel:** az első decode-nál (a scheduler foglalása után) `keret = szabad eszközmemória - tartalék` (a tartalék a `--fit-target`, `zone_vram_margin` kontextparaméter). A keret a mozgatásokkal és átméretezésekkel együtt könyvelődik (az iGPU-n a szabad memória lekérdezése nem követi a felszabadítást, ezért nem kérdezzük újra). Teszthez: `LLAMA_ZONES_HEADROOM_MIB` felülírja.
- **Adaghatáron:** a szabály a `decode()` elején fut, a következő adag előtt (szinkronizálás után), a gráf előtt. A generálás közbeni mozgatást a darabos növelés ritkítja (egy darab sok token).
- **Növelés:** ha a következő adag nem fér el (`used + n_tokens > méret`), az új méret a következő darabhatár (legfeljebb a maximum). Ha a GPU-n lévő rétegek K/V-növekménye nagyobb a keretnél, a sávból a határhoz legközelebbi GPU-s réteg lekerül (súlyok + K/V, a meglévő mozgatással), a felszabadult hely (súly + K/V) a kerethez adódik, amíg elfér vagy a sáv elfogy. Ha a sáv elfogy és a keret nem elég, a KV akkor is nő (figyelmeztetés): a kontextusból nem veszünk el, a VRAM-hiány a backend hibájaként jelentkezne.
- **Visszatétel:** ha a KV kiürül vagy jelentősen csökken (a használt cellák a méret negyede alá esnek: új beszélgetés, sorozat törlése), a KV a használathoz igazodva csökken (legalább `n_ctx_init`), majd a levett rétegek fordított sorrendben visszakerülnek, amíg a keret engedi (súlyok + K/V a pillanatnyi méretben), a host-példányból.
- **Foglalási költség:** a levett rétegek host-oldali K/V-puffere visszatétel után megmarad és a következő levételnél újrahasznosul, ha a mérete elég (RAM-költség: a sáv K/V-je a host-pufferben). A GPU-oldali foglalás nem kerülhető el (a VRAM felszabadítása a cél); ezt a VMM oldaná meg.

### 6.4 Kockázatok
- A keret könyvelése eltérhet a valódi eszközmemóriától (a compute-puffer a KV-vel nő, a backend saját foglalásai); a `--fit-target` tartalék ezt fedi. Dedikált GPU-n ellenőrizendő.
- Átméretezéskor a gráf és a scheduler foglalása újraépül (`sched_need_reserve`), ez egy-két tized másodperc lehet.
- Állapotmentés/-betöltés: a betöltött cellaszámhoz a KV előbb megnő.
- A HIP-gráfok a K/V és a súlyok címváltozása után (4. pont) itt gyakrabban jelentkeznek.

### 6.5 Megvalósítás: állapot és eltérések
- **Kész:** 6.1 (két fit, `flex_ngl_full`), 6.2 a tartalék úttal (rétegenkénti puffer, a használt sorok eszközön belüli másolása, `resize_keep`, állapotbetöltés), 6.3 (könyvelt keret, levétel a növeléskor, csökkentés és visszatétel kiürüléskor, a host-oldali K/V-puffer újrahasznosítása). Mérés: test-results.md "Automatikus háromzónás működés".
- **Nem készült el:** a VMM-út (6.2, DECISIONS.md 22); az automatikus sáv MoE-re (a kis kontextusú fit override-okat állítana: a sáv kimarad, csak a dinamikus KV működik).
- **Eltérések a tervtől:** (a) a zónalépés és a mozgatások a decode-ban az adag sorba állítása előtt futnak, és a scheduler szinkronizálását használják, nem a `synchronize()`-t (DECISIONS.md 23); (b) a szabály a máshol (állapotbetöltésnél) történt KV-növekedést is könyveli, és negatív keretnél rétegeket vesz le, mielőtt bármi más történik (DECISIONS.md 24); (c) a csökkentés a használathoz igazodó darabhatárig megy (legalább `n_ctx_init`), nem mindig a kezdőméretig.
- **Mért költség** (laptop): KV-növelés 30-94 ms (a foglalás 17-78 ms), levétel 1-5 ms rétegenként (súly + K/V), visszatétel 4-29 ms (a foglalás 1-26 ms). A foglalás maradt a nagyobb rész; a VMM ezt szüntetné meg.

## 7. Valódi használat: MoE, VMM, Flash Attention
- **MoE (DECISIONS 26):** a rugalmas sáv a két fit tenzor-szintű különbsége: egész rétegek (eszköz és K/V is mozog) és részleges rétegek (csak a zone 1-ben CPU-ra küldött GPU-tenzorok, MoE-nél a szakértők; a réteg figyelemszámítása és K/V-je a GPU-n marad). Új modellparaméter: `flex_ovr_full` (a max-kontextusú fit override-jai). Sorrend: részleges rétegek a legutolsótól lefelé, utána az egész rétegek a határ felől.
- **VMM (DECISIONS 27-29):** a ggml-cuda/HIP növelhető puffert ad (`ggml_backend_buffer_alloc_growable`, `ggml_backend_buffer_grow` a registry-n át: előre lefoglalt címtartomány, legfeljebb 64 MiB-os darabok, felszabadítás a címtartománnyal). A KV a GPU-s, nem egész-flex rétegeknél ezt használja (helyben nő, nincs átmeneti dupla foglalás), egyébként a tartalék út fut (új puffer + másolás); a zónaszabály a tartalék út csúcsát (`resize_peak`) előre felszabadítja. `LLAMA_KV_NO_VMM=1` kikapcsolja, `LLAMA_KV_VMM_SIM=1` szimulálja.
- **Flash Attention (DECISIONS 30):** a dinamikus KV a nem transzponált V-t igényli, ami `-fa auto`/`on` mellett teljesül; az FA-t nem kényszerítjük, a napló jelzi az állapotot.
- **Kockázat:** a VMM-kód csak a ROCm-buildben fordul (a laptopon nincs CUDA/HIP); a működését a NAS-on kell igazolni (SUMMARY.md 8.).

### 7.1 A VMM-leképezés érvényessége (DECISIONS 33-35)
A `hipMemSetAccess` a laptábla-frissítést csak sorba teszi. Ezért a VMM-út szétválasztja a leképezést és az első hozzáférést: a `map_to` csak leképez, a `commit` egyszer vár (KFD-korlát és/vagy `GGML_CUDA_VMM_WAIT_MS`), majd szinkron nulláz. A puffereket a `relayout` a `commit` után érinti először.

### 7.2 Megoldott akadály: a ROCm-build nem futott le (2026-11-09, megoldva 2026-11-12)
A `7a0b6086f` kód és az ollama-dualbuf `8419895f` patch-e fent van, de a `dualbuf-image` workflow nem indult el: a `csaki89/ollama` repóban ki vannak kapcsolva a GitHub Actions (`gh workflow run`: HTTP 422, "Actions has been disabled for this repository"). Ezért a `commit` lépés és a KFD-korlát HIP-en még nem fordult le, image nincs, a compose címkéje marad a `b744d77`. Teendő: az Actions visszakapcsolása, utána a workflow indítása; a várható címke `0.40.2-dualbuf-8419895`. A `vmm-stress` NAS-teszt ettől független, a checkoutból futtatható. Újabb próba az ollama-dualbuf `e1ba294a` commiton (benne a tesztprogramok image-be építése): a push nem indított buildet, a `gh workflow run` kétszer is HTTP 422-t adott ("Actions has been disabled for this repository"), bár a workflow állapota `active`. A várható címke így `0.40.2-dualbuf-e1ba294`; a Dockerfile új fordítási lépése még nem futott le. **Megoldás (2026-11-12):** a felhasználó feloldotta a GitHub fork-tiltását. Az első build (`e1ba294a`) a `vmm-stress.cpp` HIP-fordításán bukott el (`std::min`: `size_t` és `unsigned long long`), a javítás után a `b1798d9f` buildje és a smoke test sikeres. Címke: `0.40.2-dualbuf-b1798d9` (run 43316018231); a tesztek az image-ben: `/usr/lib/ollama/dualbuf-test/`.
