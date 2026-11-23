# Vulkan és ROCm: a dual-buffer, a zónák, a flex rétegek és a dinamikus KV egyezése

Felmérés a `multimodel` branchen (a45647ddb..6b19b5a28 után), módosítás előtt. Minta: a ROCm (HIP) út. Mérés: laptop, Intel Iris Xe (Vulkan, UMA/iGPU), qwen2.5-1.5B Q4_K_M, `--auto-zones --dual-buffer -fa on --load-mode none -c 16384`, `--fit-target` 5800 / 6000 / 6200 / 6500 MiB (a 2. és 3. zóna kikényszerítésére), egy ~11 000 tokenes, utána két rövid kérés. Szkript: `tools/dualbuf/vk-parity.sh`.

| elem | ROCm (minta) | Vulkan ma | eltérés |
|---|---|---|---|
| dual-buffer (CPU-s súlyok host-példánya, op offload) | `ROCm_Host` | `Vulkan_Host`, `dual-buffer: 46 weights also stored in Vulkan_Host (162.84 MiB)` | nincs |
| auto zónák (két fit, flex sáv) | `auto zones: n_ctx ... (zone 1); ... (zone 1 + 2)`, `flex band N whole layers` | ugyanezek a sorok (FIT=6000: 27 réteg az 1. zónában, 2 rétegű sáv) | nincs |
| flex rétegek mozgása a K/V-vel | `flex_move_part: layer L moved to CPU`, `flex_move: layer L K/V moved to ROCm_Host` | ugyanez `Vulkan_Host`-tal; a hosszú kérés leveszi, a rövid utáni zsugorítás visszahozza (`2 flex steps off` → `2 back`) | nincs |
| dinamikus KV helyben növelés/zsugorítás | VMM, `N/N layers in place (VMM)`, `grown/shrunk in place` | sparse binding (2. kör), `28/28 layers in place`, `grown/shrunk in place`; a flex rétegek K/V-je másolással (mint ROCm-on) | nincs |
| **a VMM-darab mérete** | futásidőben: granularitás, igazítás és a meghajtó **mért** foglalási egysége (egy minimális foglalás előtti és utáni szabad memória, kétszer); egy naplósor mutatja az összetevőket | **fix 2 MiB** (`GGML_VK_SPARSE_CHUNK_KIB`), csak a sparse blokkra kerekítve; nincs mért foglalási egység, és a darabszám nem figyeli az eszköz foglalásszám-korlátját (`maxMemoryAllocationCount`, Windowson tipikusan 4096) – nagy kontextusnál a darabok száma túllépheti | **eltér** |
| a zónakeret a mért szabad memóriából | `hipMemGetInfo` (mért) + amdgpu sysfs (`mem_info_vram_used`) összevetés | `ggml_backend_dev_memory`: `VK_EXT_memory_budget`-tel mért (heap budget − usage); **a bővítmény nélkül a heap teljes mérete**, vagyis nem mérés, és a zónakód ezt nem tudja (a keret a teljes VRAM-ból indulna); a sysfs-összevetés amdgpu-n backendtől függetlenül megy | **eltér** (tartalék és naplósor hiányzik) |
| a felszabadított memória visszajön | VMM-darab foglalásának felszabadítása, mért | `vkFreeMemory`, a budget szerint mérve (+490/560 MiB az iGPU-n) | nincs |
| `zones_state` és a többi naplósor | `zones_state ... device free: booked, driver, sysfs` | ugyanez (a `driver` a Vulkan budgetje) | költségvetés nélkül a `driver` érték félrevezető (a heap mérete) |
| tartalékág: nincs helyben növelés | `no VMM support` → másolás | `no growable (sparse) KV buffers: <hiányzó képesség>; the KV grows by copy` | nincs |
| tartalékág: nincs mért szabad memória | – (CUDA/HIP mindig mér) | nincs: a heap mérete mérésként megy | **eltér** |
| HIP-specifikus utak (KFD-korlát, „unmap keeps memory”, darabonkénti címfoglalás) | kell | nem kell (a `vkFreeMemory` azonnal visszaad) | szándékos, a backend képessége dönt |

**UMA (integrált) eszközön** a mért szabad memóriából a host-látható foglalások (pinned host-példányok, `Vulkan_Host` pufferek) is levonódnak, mert a heap közös; ezért a mért keret szűkebb lehet, mint amit a fit becsült (FIT=6200: `device memory budget -306 MiB`, a teljes sáv lemegy). Ez mérés, nem hiba; diszkrét kártyán (ROCm vagy Vulkan) a host-pufferek nem a VRAM-ból fogynak.

## Teendők (ebben a lépésben)
1. Vulkan VMM-darab futásidőben: a sparse blokk (igazítás), a meghajtó mért foglalási egysége (a budget `heapUsage` változása egy blokknyi foglalásra, kétszer) és az eszköz foglalásszám-korlátja; egy naplósor a ROCm-os mintájára; költségvetés nélkül a foglalási egység „nem mérhető”, a darab az igazításból és a korlátból jön.
2. A zónakeret mért memóriából: a backend megmondja, mér-e (`ggml_backend_dev_memory_measured` proc); ha nem, a keret a könyvelésből (az eszköz teljes memóriája − a modell eszközön lévő súlyai − a KV − a compute pufferek − a margó), és egy naplósor megnevezi a hiányzó képességet (`VK_EXT_memory_budget`). Tesztkapcsoló a letiltáshoz: `GGML_VK_DISABLE_MEMORY_BUDGET`.
3. A ROCm-os út nem változik (a proc ott nincs, a mért ágon marad).

## Eredmény
Lásd a végén (a módosítások után kitöltve).
