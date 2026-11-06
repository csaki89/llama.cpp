# Ütemterv: háromzónás súlyelhelyezés és nyitott tételek

Kiindulás: a dual-buffer ma minden CPU-n maradó repackelhető súlyhoz tart egy host-puffer másolatot (RAM-többlet), a limit és a minta csak durva szabályozás (DESIGN.md, DECISIONS.md). A cél a súlyok tudatos elhelyezése három zónában.

## 1. Háromzónás súlyelhelyezés
1. **Mindig GPU.** A forró súlyok (pl. a kimeneti réteg, az attention, a gyakran használt rétegek) fixen a VRAM-ban. Ezekhez nincs CPU-példány.
2. **Rugalmas sáv.** A súly a VRAM-ban él, de a RAM-ban is van egy **repackelt tartalék**. Ha a VRAM kell (hosszabb kontextus, nagyobb KV), a sáv rétegei a VRAM-ból kiköltöznek, és a CPU a repackelt tartalékkal számol, adatátvitel nélkül.
3. **Mindig CPU, repackelve.** A hideg súlyok csak RAM-ban, repackelve; az op offload ezekhez a dual-bufferrel (host másolat) vagy anélkül dönt.

A zónahatárok a VRAM- és RAM-keretből, a modell méretéből és a kontextushossztól jönnek; kezdetben kézi paraméter (`--dual-buffer-limit` kiterjesztése), később automatikus.

### Cél és zónák (részletesen)
Cél: a GPU-ra kerülő rész a kontextus hosszához igazodjon, **újratöltés nélkül**. Ma a KV-cache a betöltéskor a teljes beállított kontextusra foglalódik le, és a kontextusméret változása teljes újratöltést okoz.

| zóna | tartalom | hol van |
|---|---|---|
| 1. mindig GPU | ami a legnagyobb KV mellett is elfér | csak VRAM, RAM-másolat nélkül |
| 2. rugalmas sáv | ami csak rövid kontextusnál fér a GPU-ra | VRAM-ban, amíg van hely; mellette a RAM-ban repackelt tartalék, hogy a CPU azonnal átvehesse. A GPU-formájú példány a modellfájlból töltődik vissza |
| 3. mindig CPU | a hideg súlyok | csak RAM-ban, repackelve; a prompt-gyorsításhoz opcionálisan eredeti formájú másolat a host-pufferben (ez a mostani dual-buffer) |

Becslés a qwen3-coder:30b-re (32K kontextus): 1. zóna ~5,7 GB, 2. zóna ~3 GB, 3. zóna ~8,3 GB. RAM-igény ~11,3 GB (3. zóna + a 2. zóna repackelt tartaléka), a 3. zóna host-másolatával (dual-buffer) több.

### Dinamikus KV-foglalás: lehetőségek
- **(a) Átméretezés küszöböknél, másolással.** A KV a használt hossz szerint lépcsőzetesen nő; átméretezéskor új puffer, másolás, régi felszabadítása. Egyszerű, de a másolás és a csúcs-memóriahasználat (régi + új egyszerre) költsége van, és a gráf-újrahasznosítást érinti.
- **(b) Lapozott, blokkokban foglalt KV** (a vLLM PagedAttention mintájára). Rugalmas, de a figyelemszámító kernelek módosítását igényli minden backenden, ami nagy és invazív változtatás.
- **(c) Virtuális memória.** A teljes kontextus címtartománya előre le van foglalva, fizikai memória csak a használt részhez van hozzárendelve (CUDA/HIP virtuális memória API), így a KV egybefüggő marad, a kernelek változatlanok. **Ez a jelölt.** Ellenőrizendő: a ROCm támogatása gfx1030-on (RX 6700), és a Vulkan megfelelője (sparse binding), ha a megoldás backend-független marad.


### Szükséges építőkockák
- **Dinamikus KV-foglalás.** Ma a KV-cache a context létrehozásakor fix méretben foglalódik. A rugalmas sáv akkor ér valamit, ha a KV növekedhet (vagy zsugorodhat) futás közben, és a felszabaduló/lefoglalt VRAM a sávval kiegyenlíthető. Kérdés: szegmentált KV vagy újrafoglalás; a `llama_kv_cache` és a `--fit` kölcsönhatása.
- **Rétegek áthelyezése futás közben.** Egy réteg súlyai VRAM-ból RAM-ba (a repackelt tartalék már ott van, tehát csak a VRAM-példányt kell felszabadítani) és vissza (feltöltés a host-példányból). Szinkronizáció a futó gráffal, a forgalom és a kiköltöztetés ütemezése (pl. csak kérések között).
- **A számítási terv újraépítése.** Az áthelyezés megváltoztatja, melyik backendre kerül az adott réteg: a scheduler hasítását (`ggml_backend_sched_split_graph`), a gráf-újrahasznosítást és a dual-buffer ikerhozzárendelést (`ggml_backend_sched_set_weight_alt`) újra kell építeni. Mérni kell az újraépítés költségét, és a váltásokat ritkává kell tenni (hiszterézis).

### Lépések
1. Mérés: mekkora nyereséget ad egy kézi háromzónás elosztás a NAS-on (a 30/18/6 rétegsorozaton), mielőtt a dinamikát megírjuk.
2. Statikus háromzóna (zónahatárok indításkor), dinamika nélkül.
3. Dinamikus KV, majd rétegáthelyezés, majd a terv újraépítése.

## 2. Nyitott tételek
- **`--fit` és a RAM-többlet.** A `--fit` a host-puffert nem eszközmemóriaként számolja, ezért a dual-buffer RAM-többletét nem veszi figyelembe. A jelenlegi RAM-védelem (szabad RAM / cgroup) ezt részben fedi, de a `--fit` döntésébe (hány réteg menjen a GPU-ra) nem szól bele. Cél: a `--fit` a RAM-keretet is ismerje.
- **Az `ffn_` tenzorok előnyben részesítése a limitnél.** A limit ma tenzor-létrehozási sorrendben vág (korai rétegek előnyben). A 4. fázis mérése szerint az `ffn_` adja a repackelt súlyok nagy részét (265,78 / 308,18 MiB), és ez adja a legjobb ár-érték arányt. Cél: a limit alatt az `ffn_` tenzorok kapjanak elsőbbséget (két menet, vagy méret/nyereség szerinti sorrend).
- **MoE és nézetként használt súlyok.** A dual-buffert MoE-vel (`MUL_MAT_ID`, szeletelt expertek) még nem teszteltük: a nézetként átadott súlyokra nincs iker-csere (DECISIONS.md 7.). A NAS-on a qwen3-coder:30b méri meg a hatást. Ha a nézet a gyakori eset, az ikerkeresést a nézet forrástenzorára kell kiterjeszteni.
- **Több GPU tesztelése.** A host-puffer az első olyan eszközé, amelynek van (DECISIONS.md 5.). Többeszközös gépen ellenőrizendő: melyik eszköz kapja az offloadot, a tensor split és a host-puffer kapcsolata, a `--split-mode row` hatása.
- **cgroup/NAS-ellenőrzés.** A cgroup-alapú RAM-keret (DECISIONS.md 13.) valódi konténerkorláttal (`docker --memory`) még nem lett kipróbálva.

## 3. Tételek a NAS-os mérések alapján (Ollama 0.40.2, RX 6700, PCIe)
- **A prompt-sebesség a PCIe-átvitelhez kötött.** Adagonként a duplikált súlyok teljes mérete átmegy a GPU-ra (qwen3-coder:30b: ~12 GB, ~516 tok/s; gpt-oss:120b: ~43 GB, ~140 tok/s). Mérendő: nagyobb adagméret (`num_batch` / `-ub 2048`) hatása a promptra és a VRAM-ra, mert a nagyobb adag több tokenre osztja el ugyanazt az átvitelt.
- **Költségalapú op-offload döntés.** Csak akkor küldje át a súlyt, ha a másolás ideje kisebb, mint a CPU-n megspórolt számítás. MoE-nél egy 512 tokenes adagban szakértőnként átlagosan kb. 16 token jut, ami kevés az átvitel megtérüléséhez.
- **Az op offload VRAM-igénye.** Az átküldött súlyoknak helyet kell foglalni a VRAM-ban (gpt-oss:120b: számítási puffer 607 MiB, repack módban 91 MiB). Lehetőség: a GPU közvetlenül olvassa a rögzített host-memóriát, másolás nélkül.
- **gpt-oss:120b, azonos rétegelosztással:** dual a promptban +22%, a generálásban -3% (szóráson belül), +43 GB rögzített RAM áron. Nagy modelleknél a `--dual-buffer-limit` javasolt.
