# Ütemterv: háromzónás súlyelhelyezés és nyitott tételek

Kiindulás: a dual-buffer ma minden CPU-n maradó repackelhető súlyhoz tart egy host-puffer másolatot (RAM-többlet), a limit és a minta csak durva szabályozás (DESIGN.md, DECISIONS.md). A cél a súlyok tudatos elhelyezése három zónában.

## 1. Háromzónás súlyelhelyezés
1. **Mindig GPU.** A forró súlyok (pl. a kimeneti réteg, az attention, a gyakran használt rétegek) fixen a VRAM-ban. Ezekhez nincs CPU-példány.
2. **Rugalmas sáv.** A súly a VRAM-ban él, de a RAM-ban is van egy **repackelt tartalék**. Ha a VRAM kell (hosszabb kontextus, nagyobb KV), a sáv rétegei a VRAM-ból kiköltöznek, és a CPU a repackelt tartalékkal számol, adatátvitel nélkül.
3. **Mindig CPU, repackelve.** A hideg súlyok csak RAM-ban, repackelve; az op offload ezekhez a dual-bufferrel (host másolat) vagy anélkül dönt.

A zónahatárok a VRAM- és RAM-keretből, a modell méretéből és a kontextushossztól jönnek; kezdetben kézi paraméter (`--dual-buffer-limit` kiterjesztése), később automatikus.

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
