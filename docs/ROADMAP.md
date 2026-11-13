# Ütemterv: háromzónás súlyelhelyezés és nyitott tételek

## A zónák értelmezése (érvényes, az alábbi korábbi leírásokat felülírja)
1. **zóna, mindig GPU:** ami a maximális kontextus KV-cache-e mellett is elfér; csak a VRAM-ban.
2. **zóna, rugalmas sáv:** a RAM-ban mindig két formában van meg: GPU-formájú (eredeti) példány a host-pufferben és CPU-ra repackelt tartalék; amikor a réteg a GPU-n van, a VRAM-ban is.
3. **zóna, mindig CPU:** a RAM-ban CPU-ra repackelve, és ha a dual-buffer be van kapcsolva, mellette a GPU-formájú példány a prompt op offloadhoz.

Állapot: az automatikus működés (`--auto-zones`) kész: a sáv két `--fit`-futásból, dinamikus KV (átméretezés másolással), zónaváltási szabály (DESIGN-3zone.md 6); a VMM-út és a MoE-re szabott sáv hiányzik. Előzmény: a rugalmas sáv (`--flex-layers N`) kész: a GPU-s rétegek első N rétege, repackelt tartalék + host-példány a RAM-ban, futás közbeni le- és visszatétel a K/V-vel együtt; egy levett réteg úgy viselkedik, mint egy eleve CPU-s dual-bufferes réteg (DESIGN-3zone.md, test-results.md). Hiányzik: a zónaváltási szabály, a dinamikus KV, a NAS-os (ROCm, PCIe) mérés.

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
| 2. rugalmas sáv | ami csak rövid kontextusnál fér a GPU-ra | a RAM-ban mindig két formában: GPU-formájú (eredeti) példány a host-pufferben és CPU-ra repackelt tartalék; amikor a réteg a GPU-n van, a VRAM-ban is. A visszatétel a host-példányból tölt |
| 3. mindig CPU | a hideg súlyok | a RAM-ban repackelve; dual-bufferrel mellette a GPU-formájú példány a prompt op offloadhoz |

Becslés a qwen3-coder:30b-re (32K kontextus): 1. zóna ~5,7 GB, 2. zóna ~3 GB, 3. zóna ~8,3 GB. RAM-igény: 3. zóna repackelve ~8,3 GB + 2. zóna két példánya ~2 x 3 GB = ~14,3 GB; a 3. zóna host-példányával (dual-buffer) további ~8,3 GB.

### Dinamikus KV-foglalás: lehetőségek
- **(a) Átméretezés küszöböknél, másolással.** A KV a használt hossz szerint lépcsőzetesen nő; átméretezéskor új puffer, másolás, régi felszabadítása. Egyszerű, de a másolás és a csúcs-memóriahasználat (régi + új egyszerre) költsége van, és a gráf-újrahasznosítást érinti.
- **(b) Lapozott, blokkokban foglalt KV** (a vLLM PagedAttention mintájára). Rugalmas, de a figyelemszámító kernelek módosítását igényli minden backenden, ami nagy és invazív változtatás.
- **(c) Virtuális memória.** A teljes kontextus címtartománya előre le van foglalva, fizikai memória csak a használt részhez van hozzárendelve (CUDA/HIP virtuális memória API), így a KV egybefüggő marad, a kernelek változatlanok. **Ez a jelölt, a ROCm-on megvalósíthatónak bizonyult (lásd alább).** Ellenőrizendő maradt a Vulkan megfelelője (sparse binding), ha a megoldás backend-független marad.


#### A (c) változat NAS-os ellenőrzése (VMM-teszt)
Környezet: RX 6700 (gfx1031), `HSA_OVERRIDE_GFX_VERSION=10.3.0`, ROCm 7.2.1, `rocm/dev-ubuntu-24.04:7.2.1`. A tesztprogramok (`vmm_test.cpp` ... `vmm_test4.cpp`) a NAS-on vannak, a repóban nincsenek.

Mért tények:
- A VMM támogatott (`hipDeviceAttributeVirtualMemoryManagementSupported` = 1), a legkisebb egység 4 KiB.
- A címtartomány lefoglalása nem fogyaszt VRAM-ot. Több darab egymás után leképezve egybefüggő memóriaként írható.
- Egy darab leválasztása (`hipMemUnmap` + `hipMemRelease`) a címtartomány megtartásával **nem adja vissza a VRAM-ot**: a HIP visszatartja, és csak ugyanabba a címtartományba újraleképezéskor használja újra (más méretben is). Más címtartományba vagy sima `hipMalloc`-hoz nem. A VRAM csak a címtartomány felszabadításával (`hipMemAddressFree`) jön vissza.
- A `hipMemAddressReserve` kért címe csak javaslat: egymás mögé kért külön tartományok máshova kerültek.
- Egyetlen 1 GiB-os `hipMemCreate`-darab megbízhatatlan: 5-ből 5 futás GPU-laphibával összeomlott. 2, 64 és 256 MiB-os darabokkal minden futás hibátlan.
- Sávszélesség (1 GiB olvasása, 20 ismétlés): `hipMalloc` 317 GB/s, VMM 64 MiB-os és 2 MiB-os darabokkal 317 GB/s (az elméleti 320 GB/s 99%-a). Ha a VMM-foglalás ugyanabban a folyamatban egy `hipMalloc`-terület felszabadítása után jön, 303-304 GB/s (kb. -4%, minden futásban).

Következmények a tervre:
- A (c) változat **megvalósítható** ezen a kártyán.
- Darabméret legfeljebb 64-256 MiB, a súlyoknál és a KV-nél közös egységmérettel.
- A 2. zóna rétegeinek nem kell állandó cím: levételkor a súlyok a címtartományukkal együtt szabadulnak fel, visszatöltéskor új címre kerülnek, és a tenzor mutatója frissül (ez a számítási terv újraépítésével jár együtt).
- A KV egyetlen nagy címtartományban nő. Csökkentés csak új beszélgetésnél, a teljes KV-tartomány felszabadításával és újrakezdésével (darabonkénti visszaadás nem szabadít VRAM-ot).
- A -4%-os eltérést a `hipMalloc` után a valódi megvalósításban is meg kell mérni.
- Nyitott: a Vulkan megfelelője (sparse binding) nincs megvizsgálva; a VMM-teszt csak ROCm-ra vonatkozik.

#### A VMM-laphiba vizsgálata (NAS, ROCm 7.2.1, gfx1030 + HSA_OVERRIDE 10.3.0, TrueNAS kernel 6.12.33)
Mért tények:
- Ugyanaz a kód futásonként változóan működik vagy laphibát ad. A dmesg szerint a hiba pillanatában nincs bejegyzés a GPU laptáblájában (PERMISSION_FAULTS, MAPPING_ERROR, WALKER_ERROR mind 0), a `hipMemMap` és a `hipMemSetAccess` `hipSuccess`-szel tér vissza.
- Önálló teszt (granularitás, igazítás, egy nagy címtartomány, abba egymás után darabok, minden darab teljes kitöltése kernellel és visszaolvasása): várakozás nélkül 2, 4, 64 és 256 MiB-os darabbal 0/3; a `hipMemMap` + `hipMemSetAccess` utáni 1, 10 és 100 ms várakozással minden méretnél 3/3.
- Egy 9 GiB-os címtartomány, abba 24 x 256 MiB, 96 x 64 MiB és 8 x 1 GiB (6-8 GiB fizikai VRAM) várakozással hibátlan, a VRAM-foglaltság pontosan követi a leképezést, és a felszabadítás után visszaesik. (Az "egyetlen 1 GiB-os darab megbízhatatlan" korábbi megállapítás is ugyanez az időzítési hiba lehetett.)
- Kizárva: a címtartomány mérete (1 és 9 GiB egyforma), a részleges leképezés, az igazítás (4 KiB és 2 MiB egyforma), a `hipMalloc`-kal közös használat, a granularitás ismételt lekérdezése, a `hipMemGetInfo` hívása. Ahol várakozás nélkül mégis működött, ott a leképezés és az első hozzáférés közé véletlenül került ~10 ms CPU-munka.

Ok (forrásból): a `hipMemSetAccess` a ROCr-en át a DRM `GEM_VA` ioctl-t hívja, amely a laptábla-frissítést sorba teszi és nem várja meg; a KFD-s leképezés megvárja a sajátját. A megoldás és a nyitott kérdés: DECISIONS.md 33-34 (KFD-korlát + kis várakozás), a NAS-teszt: `tools/dualbuf/vmm-stress.sh`.

#### A szinkron módok NAS-mérése a valódi megvalósításon (image 0.40.2-dualbuf-7d20ed3, ROCm 7.2.1, gfx1030, `vmm-stress` gyors mód)
- A teszt érzékeny: a `none` mód az impl a és b, a pattern a, b és c esetében laphibát kap. A leképezés és az első hozzáférés között legfeljebb 0,03 ms egyéb munka van, tehát a teszt semmit nem fed el. A végső felszabadítás után a VRAM 21-116 ms alatt visszajön (DECISIONS.md 38).
- `wait` (5 ms): az impl b esetében (csökkentés, majd újranövelés) laphiba, a többi rendben. A vak várakozás 5 ms-mal sem megbízható.
- `barrier`: az impl és a pattern minden esetben rendben; a szinkron 0,18-0,35 ms, az impl b esetében egyszer 35 ms (a korlát a nagy aréna felszabadítása után a hosszú sort várta ki).
- `both`: minden rendben, ~5,3 ms, az impl b esetében 38 ms.
- Korábbi önálló mérések ugyanezen a gépen: a laptábla-frissítés a leképezés után ~30-100 µs (10-30 µs vak várakozás a nagyobb darabokkal hibázik, 100 µs elég volt); a `hipDeviceSynchronize` nem segít; egy üres kernel + `hipStreamSynchronize` sem megbízható (csak ~20 µs késleltetésként hat, 64 és 256 MiB-os darabokkal 0/5). A korlát költségének nagyobb része a `hipFree` (üresjáratban `hipMalloc` 4 MiB ~78 µs, `hipFree` ~207 µs), és a `hipFree` a teljes eszközt szinkronizálja.
- Következtetés: HIP-en az alapértelmezés a KFD-korlát várakozás nélkül; a vak várakozás csak hibakereső mód; a korlát `hipFree`-je a `commit` saját eszközszinkronja után fut (DECISIONS.md 40).

#### Később: TrueNAS 27 / 6.18-as kernel / újabb ROCm (a TrueNAS 27.0 megjelenése után megvizsgálandó)
- A 6.18-as kernel amdgpu `GEM_VA` hívása timeline syncobj-ot tud adni a laptábla-frissítés megvárására: ezzel a várakozás determinisztikus lehet, a KFD-korlát (egy friss 4 MiB-os foglalás és felszabadítás) nélkül. Ellenőrizendő, hogy a ROCr/HIP használja-e, vagy közvetlenül (libdrm) elérhető-e a mi kódunkból.
- A `rocm/dev-ubuntu-24.04:latest` (HIP 7.16) a 6.12-es kernelen VMM nélkül működik, a VMM-úton viszont az első darabnál laphiba nélkül elakad. Az elakadás okát (a régi kernel és az új ROCr összjátéka, vagy a ROCr hibája) a 6.18-as kernellel újra meg kell nézni.
- Teendő a megjelenés után: a `vmm-stress` (impl és pattern, minden mód) a 6.18-as kernelen a ROCm 7.2.1-es és az újabb ROCm-image-dzsel; ha a syncobj-os várakozás elérhető és megbízható, új szinkron mód (`syncobj`) és az alapértelmezés felülvizsgálata.

### Szükséges építőkockák
- **Dinamikus KV-foglalás.** Ma a KV-cache a context létrehozásakor fix méretben foglalódik. A rugalmas sáv akkor ér valamit, ha a KV növekedhet (vagy zsugorodhat) futás közben, és a felszabaduló/lefoglalt VRAM a sávval kiegyenlíthető. Kérdés: szegmentált KV vagy újrafoglalás; a `llama_kv_cache` és a `--fit` kölcsönhatása.
- **Rétegek áthelyezése futás közben: kész** (`--flex-layers`, DESIGN-3zone.md). Levétel: mező-csere + a GPU-puffer felszabadítása (~1 ms) + a K/V átmásolása (csak a használt cellák, DMA; a laptopon 1-20 ms, a foglalás dominál). Visszatétel: a host-példányból (6-10 ms/réteg a laptopon). A kiváltás ma csak hibakereső környezeti változóval megy; a szabály (mikor, hány réteg, hiszterézis) hiányzik.
- **A számítási terv újraépítése: kész.** A mozgatás után a gráf újraépül, a scheduler nem kell újrafoglalni; a host-példányok `weight_alt`-ként be- és kijegyződnek. A határ melletti levétel a generálási gráfot nem bontja tovább (2 hasítás 0, 1 és 2 levett réteggel).

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
