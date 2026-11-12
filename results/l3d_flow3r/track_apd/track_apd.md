# Design-0 tracking accuracy on the same 48-frame clips (clip [0, 48))

Mean over clips of the upstream WorldTrack APD / EPE (global median scale) on design-0 rows. Only clips present for every model are compared.

| Subset | Clips | APD flow3r | APD opend4rt | EPE flow3r | EPE opend4rt |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adt_mini` | 49 | 0.8056 | 0.7341 | 0.2630 | 0.2595 |
| `po_mini` | 50 | 0.3991 | 0.6897 | 0.9861 | 0.3075 |
| `pstudio_mini` | 50 | 0.4116 | 0.8105 | 0.8709 | 0.1653 |
| `ds_mini` | 50 | 0.6370 | 0.7261 | 0.4124 | 0.2958 |
