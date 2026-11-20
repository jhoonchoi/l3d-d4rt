# Q4: failure gallery

Clips chosen by the rule in `docs/l3d/STAGE1_PLAN.md` (section 4.4), from the clip APDs of the point table. Each figure: per-frame APD of every model (left) and a top-down view (x-z in camera 0, after the median scale) of up to 8 random dynamic tracks (right).

RGB overlays (`rgb_*.jpg`, frames 0/16/32/47) and MP4s come from `scripts/l3d/render_gallery.py`, which needs the clips' WorldTrack npz files; the MP4s are not committed.

| # | Subset | Sequence | Why chosen | APD OpenD4RT | APD flow3r | Figure | RGB overlay |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | `adt_mini` | `Apartment_release_clean_seq135_7` | all models fail (best APD 0.48) | 0.477 | 0.480 | [01_adt_mini_Apartment_release_clean_seq135_7.png](01_adt_mini_Apartment_release_clean_seq135_7.png) | [rgb_01_adt_mini_Apartment_release_clean_seq135_7.jpg](rgb_01_adt_mini_Apartment_release_clean_seq135_7.jpg) |
| 2 | `adt_mini` | `Apartment_release_clean_seq136_6` | flow3r ≫ OpenD4RT (Δ +0.41) | 0.417 | 0.824 | [02_adt_mini_Apartment_release_clean_seq136_6.png](02_adt_mini_Apartment_release_clean_seq136_6.png) | [rgb_02_adt_mini_Apartment_release_clean_seq136_6.jpg](rgb_02_adt_mini_Apartment_release_clean_seq136_6.jpg) |
| 3 | `adt_mini` | `Apartment_release_decoration_seq133_3` | flow3r ≫ OpenD4RT (Δ +0.40) | 0.354 | 0.753 | [03_adt_mini_Apartment_release_decoration_seq133_3.png](03_adt_mini_Apartment_release_decoration_seq133_3.png) | [rgb_03_adt_mini_Apartment_release_decoration_seq133_3.jpg](rgb_03_adt_mini_Apartment_release_decoration_seq133_3.jpg) |
| 4 | `adt_mini` | `Apartment_release_meal_seq132_2` | flow3r ≫ OpenD4RT (Δ +0.38) | 0.489 | 0.869 | [04_adt_mini_Apartment_release_meal_seq132_2.png](04_adt_mini_Apartment_release_meal_seq132_2.png) | [rgb_04_adt_mini_Apartment_release_meal_seq132_2.jpg](rgb_04_adt_mini_Apartment_release_meal_seq132_2.jpg) |
| 5 | `pstudio_mini` | `basketball_13` | all models fail (best APD 0.43) | 0.429 | 0.162 | [05_pstudio_mini_basketball_13.png](05_pstudio_mini_basketball_13.png) | [rgb_05_pstudio_mini_basketball_13.jpg](rgb_05_pstudio_mini_basketball_13.jpg) |
| 6 | `pstudio_mini` | `football_21` | OpenD4RT ≫ flow3r (Δ +0.77) | 0.903 | 0.133 | [06_pstudio_mini_football_21.png](06_pstudio_mini_football_21.png) | [rgb_06_pstudio_mini_football_21.jpg](rgb_06_pstudio_mini_football_21.jpg) |
| 7 | `pstudio_mini` | `football_3` | random reference | 0.754 | 0.602 | [07_pstudio_mini_football_3.png](07_pstudio_mini_football_3.png) | [rgb_07_pstudio_mini_football_3.jpg](rgb_07_pstudio_mini_football_3.jpg) |
| 8 | `pstudio_mini` | `softball_19` | OpenD4RT ≫ flow3r (Δ +0.67) | 0.936 | 0.267 | [08_pstudio_mini_softball_19.png](08_pstudio_mini_softball_19.png) | [rgb_08_pstudio_mini_softball_19.jpg](rgb_08_pstudio_mini_softball_19.jpg) |
| 9 | `pstudio_mini` | `softball_21` | OpenD4RT ≫ flow3r (Δ +0.80) | 0.929 | 0.125 | [09_pstudio_mini_softball_21.png](09_pstudio_mini_softball_21.png) | [rgb_09_pstudio_mini_softball_21.jpg](rgb_09_pstudio_mini_softball_21.jpg) |
| 10 | `ds_mini` | `6fae75-3_obj_source_left_1` | random reference | 0.787 | 0.693 | [10_ds_mini_6fae75-3_obj_source_left_1.png](10_ds_mini_6fae75-3_obj_source_left_1.png) | [rgb_10_ds_mini_6fae75-3_obj_source_left_1.jpg](rgb_10_ds_mini_6fae75-3_obj_source_left_1.jpg) |

## Failure tags (fill in independently, then compare)

Tags: `lost after occlusion`, `jumps to another object`, `stuck to background`, `depth drift`, `scale jump`, `other`. Fix the list after the first 5 clips.

| # | Sequence | Model | Tags (Jaehoon) | Tags (Alicia) | Notes |
| ---: | --- | --- | --- | --- | --- |
| 1 | `Apartment_release_clean_seq135_7` | OpenD4RT |  |  |  |
| 1 | `Apartment_release_clean_seq135_7` | flow3r |  |  |  |
| 2 | `Apartment_release_clean_seq136_6` | OpenD4RT |  |  |  |
| 2 | `Apartment_release_clean_seq136_6` | flow3r |  |  |  |
| 3 | `Apartment_release_decoration_seq133_3` | OpenD4RT |  |  |  |
| 3 | `Apartment_release_decoration_seq133_3` | flow3r |  |  |  |
| 4 | `Apartment_release_meal_seq132_2` | OpenD4RT |  |  |  |
| 4 | `Apartment_release_meal_seq132_2` | flow3r |  |  |  |
| 5 | `basketball_13` | OpenD4RT |  |  |  |
| 5 | `basketball_13` | flow3r |  |  |  |
| 6 | `football_21` | OpenD4RT |  |  |  |
| 6 | `football_21` | flow3r |  |  |  |
| 7 | `football_3` | OpenD4RT |  |  |  |
| 7 | `football_3` | flow3r |  |  |  |
| 8 | `softball_19` | OpenD4RT |  |  |  |
| 8 | `softball_19` | flow3r |  |  |  |
| 9 | `softball_21` | OpenD4RT |  |  |  |
| 9 | `softball_21` | flow3r |  |  |  |
| 10 | `6fae75-3_obj_source_left_1` | OpenD4RT |  |  |  |
| 10 | `6fae75-3_obj_source_left_1` | flow3r |  |  |  |
