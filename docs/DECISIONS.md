# Döntések (éjszakai önálló munka)

Formátum: döntés, indok. A felhasználó nem volt elérhető, ezért ezek saját mérlegelésen alapulnak.

1. **Az iker-hozzárendelés a schedulerben él** (`ggml_backend_sched_set_weight_alt`), nem a `ggml_tensor`-ban. Indok: a `ggml_tensor` bővítése minden backendet érint; a scheduler-oldali map lokális, és backend-független. (Részletek: DESIGN.md 2.2 és 3.)
2. **Dual-buffer csak `--load-mode none` mellett működik.** mmap esetén a host-puffer választását a meglévő kód úgyis felülírja (`llama-model-loader.cpp:1282`), így nem lenne iker. Figyelmeztetéssel kikapcsol, a viselkedés ilyenkor az eredeti.
3. **`--dual-buffer` és `--no-host` együtt:** a dual-buffer nyer. Az elsődleges lista úgyis host-puffer nélkül épül, az iker a host-puffert használja. Indok: a `--no-host` jelentése "ne legyen host-puffer", a dual-buffer pont kettőt akar, ezért az explicit dual-buffer a konkrétabb kérés. Naplózva.
4. **`--no-repack` mellett a dual-buffer kikapcsol** (nincs mit duplikálni), figyelmeztetéssel.
5. **Az iker puffertípusa:** az első olyan eszköz host-puffere, amelynek van (ugyanaz a logika, mint a mai `make_cpu_buft_list()`). Indok: nem kell új konfiguráció, többeszközös gépen is determinisztikus.
6. **Az ikertenzorok nem kerülnek a `tensors_by_name`-be.** Indok: a név szerinti keresés az első egyezést adja, és nem lehet garantálni, melyik context jön előbb.
7. **Nézetként használt súlyokra nincs iker-csere** (a mai viselkedés marad). Indok: minimális változtatás; a dense modellek közvetlenül használják a súlyt.
8. **Tesztszkript:** a `bench.sh` marad bash. A kimenet-összehasonlítás md5-tel, a sebesség 3 ismétlés átlagával (szórással).
9. **Korlátozás (4. fázis):** a RAM-keret a másolatok összmérete MiB-ban, tenzor-létrehozási sorrendben vág (a korai rétegek előnyben). Indok: egyszerű, determinisztikus, a betöltő egy menetben dönt. Tenzorszűrés: regex a tenzornévre (`regex_search`, mint a meglévő `--override-tensor`). Mindkettő csak `--dual-buffer` mellett hat; a `llama_model_params`-ba két új mező került (`dual_buffer_pattern`, `dual_buffer_limit_mib`), a bool mezők közé nem, hogy a szerkezet bool-blokkja egyben maradjon.
10. **Az `auto` betöltési mód nem mmap ezen a gépen,** ezért ott a `--dual-buffer` explicit `--load-mode none` nélkül is aktív. Explicit `--load-mode mmap` esetén figyelmeztetéssel kikapcsol. Nem módosítottam az `auto` logikát.
