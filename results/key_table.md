**Funzioni del paper**: valutazioni per raggiungere f − f* ≤ 1e-8 (mediana sui run riusciti; tra parentesi run riusciti/totali)

| problema | d | budget | PBO orig. | **PBO+** | CMA-ES | lq-CMA-ES |
|---|---|---|---|---|---|---|
| sphere_2d | 2 | 300 | 280 (8/10) | 16 (10/10) | 246 (8/10) | 17 (10/10) |
| branin_2d | 2 | 300 | 295 (3/10) | 34 (9/10) | 272 (6/10) | 72 (10/10) |
| griewank_2d | 2 | 300 | – (0/10), med f=0.03 | – (0/10), med f=0.02 | – (0/10), med f=0.02 | – (0/10), med f=0.02 |
| himmelblau | 2 | 210 | – (0/10), med f=0.003 | 70 (10/10) | – (0/10), med f=0.02 | 80 (10/10) |
| parabola_triangle | 2 | 300 | 185 (10/10) | 26 (10/10) | 190 (10/10) | 46 (10/10) |
| rosenbrock_2d | 2 | 900 | – (0/10), med f=1e-07 | 151 (10/10) | 452 (10/10) | 153 (10/10) |
| rosenbrock_5d | 5 | 2400 | – (0/10), med f=1e-06 | 414 (10/10) | 1805 (9/10) | 403 (9/10) |
| rosenbrock_10d | 10 | 6000 | – (0/10), med f=4e-05 | 1327 (8/10) | 5192 (10/10) | 1357 (9/10) |

**Funzioni di controllo (mai usate per la taratura)**: valutazioni per raggiungere f − f* ≤ 1e-8 (mediana sui run riusciti; tra parentesi run riusciti/totali)

| problema | d | budget | PBO orig. | **PBO+** | CMA-ES | lq-CMA-ES |
|---|---|---|---|---|---|---|
| rot_ellipsoid_5d | 5 | 1200 | – (0/5), med f=1e+02 | – (0/10), med f=9 | – (0/10), med f=2 | 667 (7/10) |
| rot_ellipsoid_10d | 10 | 3000 | – (0/5), med f=1e+02 | – (0/10), med f=1e+02 | – (0/10), med f=6e+01 | 2721 (1/10) |
| diffpow_8d | 8 | 1500 | 1161 (5/5) | 424 (9/10) | 1086 (10/10) | 447 (10/10) |
| levy_5d | 5 | 1200 | – (0/5), med f=3e-07 | 246 (8/10) | 811 (8/10) | 258 (4/10) |
| styblinski_4d | 4 | 800 | – (0/5), med f=1e+01 | 291 (3/10) | – (0/10), med f=2e+01 | 199 (2/10) |
| ackley_5d | 5 | 1200 | – (0/5), med f=0.0002 | 410 (10/10) | – (0/10), med f=9e-08 | 625 (9/10) |
| rastrigin_3d | 3 | 900 | – (0/5), med f=5 | 747 (1/10) | 695 (1/10) | – (0/10), med f=3 |
| rosenbrock_20d | 20 | 12000 | n/a | 5340 (9/10) | – (0/10), med f=3 | 5452 (10/10) |

