# v1 (deployed) vs v2 (retrained) on the same held-out windows

## public_heldout_all

n = 27352

| | v1 | v2 |
|---|---|---|
| accuracy | 0.3362 | 0.779 |
| macro F1 (classes present) | 0.197 | 0.796 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 5908 | 0.0 / 0.0 / 0.0 | 0.707 / 0.473 / 0.567 |
| Jogging | 1181 | 0.053 / 0.393 / 0.093 | 0.888 / 0.886 / 0.887 |
| Sports | 4893 | 0.0 / 0.0 / 0.0 | 0.798 / 0.772 / 0.785 |
| Stairs | 4571 | 0.0 / 0.0 / 0.0 | 0.846 / 0.928 / 0.885 |
| Still | 7340 | 0.592 / 0.963 / 0.734 | 0.711 / 0.895 / 0.793 |
| Walking | 3459 | 0.282 / 0.481 / 0.355 | 0.889 / 0.833 / 0.86 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Hand Activity | 0 | 791 | 0 | 314 | 4418 | 385 |
| Jogging | 0 | 464 | 0 | 1 | 1 | 715 |
| Sports | 0 | 3154 | 0 | 322 | 444 | 973 |
| Stairs | 0 | 2430 | 0 | 0 | 1 | 2140 |
| Still | 0 | 160 | 0 | 81 | 7069 | 30 |
| Walking | 0 | 1795 | 0 | 0 | 0 | 1664 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Hand Activity | 2794 | 9 | 564 | 33 | 2489 | 19 |
| Jogging | 16 | 1046 | 47 | 33 | 2 | 37 |
| Sports | 482 | 39 | 3776 | 298 | 172 | 126 |
| Stairs | 8 | 20 | 128 | 4240 | 0 | 175 |
| Still | 649 | 3 | 103 | 11 | 6571 | 3 |
| Walking | 3 | 61 | 114 | 398 | 2 | 2881 |

## public_heldout_wisdm

n = 15357

| | v1 | v2 |
|---|---|---|
| accuracy | 0.242 | 0.6348 |
| macro F1 (classes present) | 0.1792 | 0.6705 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 5908 | 0.0 / 0.0 / 0.0 | 0.741 / 0.473 / 0.577 |
| Jogging | 1181 | 0.12 / 0.393 / 0.184 | 0.889 / 0.886 / 0.888 |
| Sports | 3543 | 0.0 / 0.0 / 0.0 | 0.729 / 0.717 / 0.723 |
| Stairs | 1181 | 0.0 / 0.0 / 0.0 | 0.58 / 0.757 / 0.657 |
| Still | 2363 | 0.313 / 0.899 / 0.464 | 0.4 / 0.751 / 0.522 |
| Walking | 1181 | 0.275 / 0.954 / 0.427 | 0.734 / 0.593 / 0.656 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Hand Activity | 0 | 791 | 0 | 314 | 4418 | 385 |
| Jogging | 0 | 464 | 0 | 1 | 1 | 715 |
| Sports | 0 | 2134 | 0 | 228 | 254 | 927 |
| Stairs | 0 | 269 | 0 | 0 | 1 | 911 |
| Still | 0 | 146 | 0 | 62 | 2125 | 30 |
| Walking | 0 | 54 | 0 | 0 | 0 | 1127 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Hand Activity | 2794 | 9 | 564 | 33 | 2489 | 19 |
| Jogging | 16 | 1046 | 47 | 33 | 2 | 37 |
| Sports | 479 | 37 | 2541 | 258 | 165 | 63 |
| Stairs | 8 | 20 | 127 | 894 | 0 | 132 |
| Still | 469 | 3 | 103 | 11 | 1774 | 3 |
| Walking | 3 | 61 | 105 | 312 | 0 | 700 |

## public_heldout_heterogeneity

n = 9656

| | v1 | v2 |
|---|---|---|
| accuracy | 0.4379 | 0.9599 |
| macro F1 (classes present) | 0.3145 | 0.9642 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Jogging | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Sports | 1350 | 0.0 / 0.0 / 0.0 | 0.993 / 0.915 / 0.952 |
| Stairs | 2710 | 0.0 / 0.0 / 0.0 | 0.963 / 0.985 / 0.974 |
| Still | 3710 | 0.951 / 1.0 / 0.975 | 0.997 / 0.957 / 0.977 |
| Walking | 1886 | 0.291 / 0.275 / 0.283 | 0.947 / 0.961 / 0.954 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Sports | 0 | 1020 | 0 | 94 | 190 | 46 |
| Stairs | 0 | 1494 | 0 | 0 | 0 | 1216 |
| Still | 0 | 1 | 0 | 0 | 3709 | 0 |
| Walking | 0 | 1367 | 0 | 0 | 0 | 519 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Sports | 3 | 2 | 1235 | 40 | 7 | 63 |
| Stairs | 0 | 0 | 1 | 2670 | 0 | 39 |
| Still | 159 | 0 | 0 | 0 | 3551 | 0 |
| Walking | 0 | 0 | 8 | 63 | 2 | 1813 |

## public_heldout_uci

n = 2339

| | v1 | v2 |
|---|---|---|
| accuracy | 0.5357 | 0.9791 |
| macro F1 (classes present) | 0.3574 | 0.9785 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Jogging | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Sports | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Stairs | 680 | 0.0 / 0.0 / 0.0 | 0.967 / 0.994 / 0.98 |
| Still | 1267 | 1.0 / 0.975 / 0.987 | 1.0 / 0.983 / 0.992 |
| Walking | 392 | 0.581 / 0.046 / 0.085 | 0.989 / 0.939 / 0.963 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Stairs | 0 | 667 | 0 | 0 | 0 | 13 |
| Still | 0 | 13 | 0 | 19 | 1235 | 0 |
| Walking | 0 | 374 | 0 | 0 | 0 | 18 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Stairs | 0 | 0 | 0 | 676 | 0 | 4 |
| Still | 21 | 0 | 0 | 0 | 1246 | 0 |
| Walking | 0 | 0 | 1 | 23 | 0 | 368 |

## phone_heldout_stairs_session

n = 117

| | v1 | v2 |
|---|---|---|
| accuracy | 0.0 | 0.4615 |
| macro F1 (classes present) | 0.0 | 0.6316 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Jogging | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Sports | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Stairs | 117 | 0.0 / 0.0 / 0.0 | 1.0 / 0.462 / 0.632 |
| Still | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Walking | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Stairs | 0 | 62 | 0 | 0 | 5 | 50 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Stairs | 6 | 4 | 17 | 54 | 6 | 30 |

## phone_heldout_walking2

n = 48

| | v1 | v2 |
|---|---|---|
| accuracy | 1.0 | 0.875 |
| macro F1 (classes present) | 1.0 | 0.9333 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Jogging | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Sports | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Stairs | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Still | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Walking | 48 | 1.0 / 1.0 / 1.0 | 1.0 / 0.875 / 0.933 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Walking | 0 | 0 | 0 | 0 | 0 | 48 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Walking | 0 | 1 | 1 | 4 | 0 | 42 |

## phone_seen_in_training

n = 230

| | v1 | v2 |
|---|---|---|
| accuracy | 1.0 | 1.0 |
| macro F1 (classes present) | 1.0 | 1.0 |

| class | support | v1 P / R / F1 | v2 P / R / F1 |
|---|---|---|---|
| Hand Activity | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Jogging | 64 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Sports | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Stairs | 0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
| Still | 103 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| Walking | 63 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |

v1 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Jogging | 0 | 64 | 0 | 0 | 0 | 0 |
| Still | 0 | 0 | 0 | 0 | 103 | 0 |
| Walking | 0 | 0 | 0 | 0 | 0 | 63 |

v2 confusion (rows = true, cols = predicted)

| true \ pred | Hand Activity | Jogging | Sports | Stairs | Still | Walking |
|---|---|---|---|---|---|---|
| Jogging | 0 | 64 | 0 | 0 | 0 | 0 |
| Still | 0 | 0 | 0 | 0 | 103 | 0 |
| Walking | 0 | 0 | 0 | 0 | 0 | 63 |

## Dashboard pipeline (rules + smoothing) on phone recordings

| recording | label | n | v1 accuracy | v1 predictions | v2 accuracy | v2 predictions |
|---|---|---|---|---|---|---|
| stairs1 | Stairs | 63 | 0.0 | {'Jogging': 50, 'Uncertain': 6, 'Walking': 7} | 0.8413 | {'Sports': 3, 'Stairs': 53, 'Uncertain': 7} |
| stairs2 | Stairs | 54 | 0.0 | {'Jogging': 3, 'Still': 9, 'Uncertain': 6, 'Walking': 36} | 0.1667 | {'Hand Activity': 5, 'Stairs': 9, 'Still': 5, 'Uncertain': 12, 'Walking': 23} |
| walking2 | Walking | 48 | 1.0 | {'Walking': 48} | 0.8125 | {'Jogging': 5, 'Stairs': 4, 'Walking': 39} |
| still | Still | 103 | 1.0 | {'Still': 103} | 1.0 | {'Still': 103} |
| jogging | Jogging | 64 | 1.0 | {'Jogging': 64} | 1.0 | {'Jogging': 64} |
| walking1 | Walking | 63 | 1.0 | {'Walking': 63} | 1.0 | {'Walking': 63} |

## Model size and latency

| | parameters | median ms | p95 ms |
|---|---|---|---|
| v1 | 840647 | 19.45 | 20.59 |
| v2 | 840647 | 19.88 | 21.9 |
