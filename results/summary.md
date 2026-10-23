### Benchmarks of the PBO paper (same budgets and λ as the original json files)

| problem | algorithm | runs | median final | q25 | q75 | evals to 1e-3 (median, solved) | evals to 1e-8 (median, solved) |
|---|---|---|---|---|---|---|---|
| sphere_2d | **PBO (original code)** | 10 | 2.4e-09 | 6.1e-10 | 7.7e-09 | 113 (10/10) | 280 (8/10) |
| sphere_2d | **PBO+ (full)** | 10 | 3.9e-31 | 3.9e-31 | 3.9e-31 | 16 (10/10) | 16 (10/10) |
| sphere_2d | PBO+ w/o actor step | 10 | 1.3e-30 | 9.9e-31 | 4.3e-30 | 28 (10/10) | 53 (10/10) |
| sphere_2d | PBO+ w/o incremental critic | 10 | 3.9e-31 | 3.9e-31 | 3.9e-31 | 22 (10/10) | 23 (10/10) |
| sphere_2d | PBO+ w/o critic | 10 | 1.5e-09 | 9.3e-11 | 4.5e-09 | 108 (10/10) | 249 (9/10) |
| sphere_2d | natural-gradient PBO (xNES) | 10 | 9.2e-09 | 5.2e-09 | 1.0e-07 | 132 (10/10) | 296 (5/10) |
| sphere_2d | xNES + IS reuse | 10 | 2.1e-10 | 6.8e-11 | 3.4e-10 | 98 (10/10) | 246 (10/10) |
| sphere_2d | CMA-ES (pycma) | 10 | 7.6e-10 | 7.4e-11 | 7.2e-09 | 105 (10/10) | 246 (8/10) |
| sphere_2d | lq-CMA-ES (pycma) | 10 | 2.7e-28 | 6.7e-30 | 6.9e-28 | 16 (10/10) | 17 (10/10) |
| branin_2d | **PBO (original code)** | 10 | 1.9e-08 | 4.5e-09 | 6.5e-08 | 150 (10/10) | 295 (3/10) |
| branin_2d | **PBO+ (full)** | 10 | -2.2e-16 | -2.2e-16 | -2.2e-16 | 28 (9/10) | 34 (9/10) |
| branin_2d | PBO+ w/o actor step | 10 | 2.3e-10 | 7.1e-12 | 7.1e-07 | 108 (10/10) | 233 (7/10) |
| branin_2d | PBO+ w/o incremental critic | 10 | -2.2e-16 | -2.2e-16 | -2.2e-16 | 48 (10/10) | 58 (10/10) |
| branin_2d | PBO+ w/o critic | 10 | 9.7e-08 | 7.9e-09 | 8.3e-07 | 165 (10/10) | 292 (3/10) |
| branin_2d | natural-gradient PBO (xNES) | 10 | 4.2e-07 | 2.4e-07 | 1.7e-06 | 165 (10/10) | – (0/10) |
| branin_2d | xNES + IS reuse | 10 | 1.7e-09 | 5.8e-10 | 8.1e-09 | 144 (10/10) | 268 (8/10) |
| branin_2d | CMA-ES (pycma) | 10 | 2.7e-09 | 5.0e-10 | 7.1e-08 | 141 (10/10) | 272 (6/10) |
| branin_2d | lq-CMA-ES (pycma) | 10 | -2.2e-16 | -2.2e-16 | -2.2e-16 | 43 (10/10) | 72 (10/10) |
| griewank_2d | **PBO (original code)** | 10 | 2.7e-02 | 1.1e-02 | 3.0e-02 | 164 (1/10) | – (0/10) |
| griewank_2d | **PBO+ (full)** | 10 | 2.0e-02 | 8.0e-03 | 2.7e-02 | 272 (1/10) | – (0/10) |
| griewank_2d | PBO+ w/o actor step | 10 | 2.5e-02 | 1.6e-02 | 3.0e-02 | – (0/10) | – (0/10) |
| griewank_2d | PBO+ w/o incremental critic | 10 | 1.4e-02 | 9.1e-03 | 2.7e-02 | – (0/10) | – (0/10) |
| griewank_2d | PBO+ w/o critic | 10 | 2.5e-02 | 9.9e-03 | 3.0e-02 | – (0/10) | – (0/10) |
| griewank_2d | natural-gradient PBO (xNES) | 10 | 2.8e-02 | 2.2e-02 | 3.0e-02 | – (0/10) | – (0/10) |
| griewank_2d | xNES + IS reuse | 10 | 2.8e-02 | 2.7e-02 | 3.0e-02 | – (0/10) | – (0/10) |
| griewank_2d | CMA-ES (pycma) | 10 | 1.8e-02 | 9.9e-03 | 2.7e-02 | – (0/10) | – (0/10) |
| griewank_2d | lq-CMA-ES (pycma) | 10 | 1.8e-02 | 9.9e-03 | 2.9e-02 | – (0/10) | – (0/10) |
| himmelblau | **PBO (original code)** | 10 | 3.3e-03 | 5.2e-04 | 6.4e-03 | 181 (3/10) | – (0/10) |
| himmelblau | **PBO+ (full)** | 10 | 3.9e-31 | 0.0e+00 | 7.1e-30 | 62 (10/10) | 70 (10/10) |
| himmelblau | PBO+ w/o actor step | 10 | 1.3e-04 | 1.5e-07 | 6.7e-03 | 68 (6/10) | 80 (3/10) |
| himmelblau | PBO+ w/o incremental critic | 10 | 3.5e-30 | 0.0e+00 | 7.1e-30 | 84 (10/10) | 99 (10/10) |
| himmelblau | PBO+ w/o critic | 10 | 2.0e-02 | 7.2e-03 | 1.0e-01 | – (0/10) | – (0/10) |
| himmelblau | natural-gradient PBO (xNES) | 10 | 7.6e-02 | 2.6e-02 | 2.3e-01 | – (0/10) | – (0/10) |
| himmelblau | xNES + IS reuse | 10 | 1.1e-02 | 4.1e-03 | 7.0e-02 | – (0/10) | – (0/10) |
| himmelblau | CMA-ES (pycma) | 10 | 1.6e-02 | 8.4e-04 | 1.1e-01 | 188 (3/10) | – (0/10) |
| himmelblau | lq-CMA-ES (pycma) | 10 | 1.3e-15 | 3.6e-16 | 8.4e-15 | 57 (10/10) | 80 (10/10) |
| parabola_triangle | **PBO (original code)** | 10 | 4.2e-11 | 1.6e-11 | 1.6e-10 | 60 (10/10) | 185 (10/10) |
| parabola_triangle | **PBO+ (full)** | 10 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 17 (10/10) | 26 (10/10) |
| parabola_triangle | PBO+ w/o actor step | 10 | 2.7e-14 | 4.7e-16 | 1.9e-13 | 15 (10/10) | 95 (10/10) |
| parabola_triangle | PBO+ w/o incremental critic | 10 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 19 (10/10) | 36 (10/10) |
| parabola_triangle | PBO+ w/o critic | 10 | 9.4e-10 | 6.6e-11 | 2.5e-09 | 35 (10/10) | 228 (8/10) |
| parabola_triangle | natural-gradient PBO (xNES) | 10 | 1.3e-09 | 2.5e-10 | 3.0e-09 | 29 (10/10) | 248 (9/10) |
| parabola_triangle | xNES + IS reuse | 10 | 3.2e-12 | 2.4e-12 | 7.0e-12 | 34 (10/10) | 214 (9/10) |
| parabola_triangle | CMA-ES (pycma) | 10 | 1.2e-11 | 1.5e-12 | 6.5e-11 | 41 (10/10) | 190 (10/10) |
| parabola_triangle | lq-CMA-ES (pycma) | 10 | 8.0e-16 | 2.9e-16 | 1.5e-15 | 16 (10/10) | 46 (10/10) |
| rosenbrock_2d | **PBO (original code)** | 10 | 1.1e-07 | 9.6e-08 | 2.9e-07 | 485 (9/10) | – (0/10) |
| rosenbrock_2d | **PBO+ (full)** | 10 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 137 (10/10) | 151 (10/10) |
| rosenbrock_2d | PBO+ w/o actor step | 10 | 0.0e+00 | 0.0e+00 | 2.0e-31 | 189 (10/10) | 223 (10/10) |
| rosenbrock_2d | PBO+ w/o incremental critic | 10 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 146 (10/10) | 167 (10/10) |
| rosenbrock_2d | PBO+ w/o critic | 10 | 2.7e-17 | 2.6e-19 | 7.9e-15 | 445 (10/10) | 598 (10/10) |
| rosenbrock_2d | natural-gradient PBO (xNES) | 10 | 5.2e-21 | 2.7e-22 | 2.1e-19 | 200 (10/10) | 421 (9/10) |
| rosenbrock_2d | xNES + IS reuse | 10 | 6.9e-28 | 2.8e-28 | 1.3e-26 | 223 (10/10) | 343 (10/10) |
| rosenbrock_2d | CMA-ES (pycma) | 10 | 6.7e-24 | 8.2e-25 | 1.2e-22 | 285 (10/10) | 452 (10/10) |
| rosenbrock_2d | lq-CMA-ES (pycma) | 10 | 3.7e-16 | 2.0e-16 | 1.2e-15 | 125 (10/10) | 153 (10/10) |
| rosenbrock_5d | **PBO (original code)** | 10 | 9.8e-07 | 7.9e-07 | 1.7e-06 | 1468 (10/10) | – (0/10) |
| rosenbrock_5d | **PBO+ (full)** | 10 | 3.2e-29 | 0.0e+00 | 4.4e-29 | 386 (10/10) | 414 (10/10) |
| rosenbrock_5d | PBO+ w/o actor step | 10 | 6.0e-29 | 4.5e-29 | 7.7e-29 | 391 (10/10) | 443 (9/10) |
| rosenbrock_5d | PBO+ w/o incremental critic | 10 | 3.2e-29 | 1.7e-29 | 4.5e-29 | 669 (10/10) | 706 (9/10) |
| rosenbrock_5d | PBO+ w/o critic | 10 | 1.9e-08 | 2.6e-10 | 2.8e-07 | 1875 (8/10) | 2301 (5/10) |
| rosenbrock_5d | natural-gradient PBO (xNES) | 10 | 4.2e-06 | 3.4e-07 | 1.6e-03 | 1846 (6/10) | 2328 (1/10) |
| rosenbrock_5d | xNES + IS reuse | 10 | 1.0e-08 | 2.1e-09 | 1.4e-07 | 1802 (9/10) | 2321 (5/10) |
| rosenbrock_5d | CMA-ES (pycma) | 10 | 4.5e-15 | 9.0e-18 | 6.4e-14 | 1452 (9/10) | 1805 (9/10) |
| rosenbrock_5d | lq-CMA-ES (pycma) | 10 | 5.8e-15 | 4.0e-15 | 8.0e-15 | 362 (9/10) | 403 (9/10) |
| rosenbrock_10d | **PBO (original code)** | 10 | 4.4e-05 | 1.8e-05 | 8.8e-05 | 4892 (9/10) | – (0/10) |
| rosenbrock_10d | **PBO+ (full)** | 10 | 1.1e-28 | 8.7e-29 | 7.3e-15 | 1247 (8/10) | 1327 (8/10) |
| rosenbrock_10d | PBO+ w/o actor step | 10 | 1.6e-28 | 1.3e-28 | 2.6e-11 | 1425 (9/10) | 1547 (9/10) |
| rosenbrock_10d | PBO+ w/o incremental critic | 10 | 1.4e-28 | 1.1e-28 | 1.0e-12 | 2395 (10/10) | 2486 (10/10) |
| rosenbrock_10d | PBO+ w/o critic | 10 | 2.6e-05 | 9.6e-07 | 1.5e-04 | 5494 (9/10) | – (0/10) |
| rosenbrock_10d | natural-gradient PBO (xNES) | 10 | 3.3e-01 | 1.5e-01 | 1.8e+00 | – (0/10) | – (0/10) |
| rosenbrock_10d | xNES + IS reuse | 10 | 8.0e-01 | 3.6e-01 | 1.0e+00 | – (0/10) | – (0/10) |
| rosenbrock_10d | CMA-ES (pycma) | 10 | 7.6e-14 | 1.5e-15 | 9.0e-13 | 4436 (10/10) | 5192 (10/10) |
| rosenbrock_10d | lq-CMA-ES (pycma) | 10 | 1.4e-14 | 1.1e-14 | 2.2e-14 | 1229 (9/10) | 1357 (9/10) |

### Hold-out functions (never used to design or tune PBO+)

| problem | algorithm | runs | median final | q25 | q75 | evals to 1e-3 (median, solved) | evals to 1e-8 (median, solved) |
|---|---|---|---|---|---|---|---|
| rot_ellipsoid_5d | **PBO (original code)** | 5 | 1.0e+02 | 8.6e+00 | 1.5e+02 | – (0/5) | – (0/5) |
| rot_ellipsoid_5d | **PBO+ (full)** | 10 | 8.7e+00 | 3.2e+00 | 2.2e+01 | – (0/10) | – (0/10) |
| rot_ellipsoid_5d | PBO+ w/o critic | 10 | 5.0e+00 | 3.6e+00 | 3.6e+01 | – (0/10) | – (0/10) |
| rot_ellipsoid_5d | natural-gradient PBO (xNES) | 10 | 1.0e+02 | 8.4e+01 | 1.5e+02 | – (0/10) | – (0/10) |
| rot_ellipsoid_5d | CMA-ES (pycma) | 10 | 2.2e+00 | 4.4e-01 | 7.2e+00 | 1137 (2/10) | – (0/10) |
| rot_ellipsoid_5d | lq-CMA-ES (pycma) | 10 | 4.0e-17 | 1.9e-18 | 8.9e-01 | 667 (7/10) | 667 (7/10) |
| rot_ellipsoid_10d | **PBO (original code)** | 5 | 1.1e+02 | 6.0e+01 | 1.2e+02 | – (0/5) | – (0/5) |
| rot_ellipsoid_10d | **PBO+ (full)** | 10 | 1.1e+02 | 3.4e+01 | 1.7e+02 | – (0/10) | – (0/10) |
| rot_ellipsoid_10d | PBO+ w/o critic | 10 | 1.2e+02 | 4.8e+01 | 3.4e+02 | – (0/10) | – (0/10) |
| rot_ellipsoid_10d | natural-gradient PBO (xNES) | 10 | 5.0e+02 | 2.8e+02 | 7.8e+02 | – (0/10) | – (0/10) |
| rot_ellipsoid_10d | CMA-ES (pycma) | 10 | 6.5e+01 | 5.3e+01 | 1.3e+02 | – (0/10) | – (0/10) |
| rot_ellipsoid_10d | lq-CMA-ES (pycma) | 10 | 6.0e+00 | 1.9e+00 | 1.6e+01 | 2721 (1/10) | 2721 (1/10) |
| diffpow_8d | **PBO (original code)** | 5 | 9.3e-10 | 2.0e-10 | 1.2e-09 | 484 (5/5) | 1161 (5/5) |
| diffpow_8d | **PBO+ (full)** | 10 | 4.1e-28 | 2.5e-29 | 5.8e-27 | 169 (10/10) | 424 (9/10) |
| diffpow_8d | PBO+ w/o critic | 10 | 3.5e-08 | 1.5e-08 | 4.7e-08 | 359 (10/10) | 1358 (1/10) |
| diffpow_8d | natural-gradient PBO (xNES) | 10 | 6.7e-05 | 2.8e-05 | 7.9e-05 | 1066 (10/10) | – (0/10) |
| diffpow_8d | CMA-ES (pycma) | 10 | 8.9e-11 | 5.6e-11 | 2.4e-10 | 369 (10/10) | 1086 (10/10) |
| diffpow_8d | lq-CMA-ES (pycma) | 10 | 1.9e-12 | 1.5e-12 | 3.6e-12 | 152 (10/10) | 447 (10/10) |
| levy_5d | **PBO (original code)** | 5 | 3.2e-07 | 9.1e-08 | 5.2e-02 | 636 (3/5) | – (0/5) |
| levy_5d | **PBO+ (full)** | 10 | 2.2e-30 | 2.2e-30 | 2.2e-30 | 218 (8/10) | 246 (8/10) |
| levy_5d | PBO+ w/o critic | 10 | 5.0e-12 | 2.8e-13 | 1.7e+00 | 427 (7/10) | 818 (7/10) |
| levy_5d | natural-gradient PBO (xNES) | 10 | 4.9e-05 | 3.8e-05 | 1.2e-04 | 951 (10/10) | – (0/10) |
| levy_5d | CMA-ES (pycma) | 10 | 8.1e-14 | 2.1e-14 | 6.7e-13 | 433 (8/10) | 811 (8/10) |
| levy_5d | lq-CMA-ES (pycma) | 10 | 9.0e-02 | 6.6e-15 | 8.2e-01 | 223 (4/10) | 258 (4/10) |
| styblinski_4d | **PBO (original code)** | 5 | 1.4e+01 | 1.4e+01 | 2.8e+01 | 494 (1/5) | – (0/5) |
| styblinski_4d | **PBO+ (full)** | 10 | 1.4e+01 | 2.7e+00 | 1.7e+01 | 274 (3/10) | 291 (3/10) |
| styblinski_4d | PBO+ w/o critic | 10 | 1.4e+01 | 1.4e+01 | 2.8e+01 | 488 (2/10) | 692 (1/10) |
| styblinski_4d | natural-gradient PBO (xNES) | 10 | 1.2e-02 | 3.1e-03 | 1.3e+01 | 706 (2/10) | – (0/10) |
| styblinski_4d | CMA-ES (pycma) | 10 | 2.1e+01 | 1.4e+01 | 2.8e+01 | – (0/10) | – (0/10) |
| styblinski_4d | lq-CMA-ES (pycma) | 10 | 2.8e+01 | 1.4e+01 | 2.8e+01 | 184 (2/10) | 199 (2/10) |
| ackley_5d | **PBO (original code)** | 5 | 2.5e-04 | 1.0e-04 | 4.5e-04 | 826 (5/5) | – (0/5) |
| ackley_5d | **PBO+ (full)** | 10 | 1.3e-14 | 7.5e-15 | 1.7e-14 | 216 (10/10) | 410 (10/10) |
| ackley_5d | PBO+ w/o critic | 10 | 3.9e-07 | 9.5e-08 | 4.9e-07 | 644 (10/10) | – (0/10) |
| ackley_5d | natural-gradient PBO (xNES) | 10 | 5.8e-03 | 4.8e-03 | 6.9e-03 | – (0/10) | – (0/10) |
| ackley_5d | CMA-ES (pycma) | 10 | 9.4e-08 | 4.0e-08 | 2.0e-07 | 572 (9/10) | – (0/10) |
| ackley_5d | lq-CMA-ES (pycma) | 10 | 1.4e-10 | 1.2e-10 | 1.5e-10 | 206 (9/10) | 625 (9/10) |
| rastrigin_3d | **PBO (original code)** | 5 | 4.9e+00 | 3.5e+00 | 5.0e+00 | – (0/5) | – (0/5) |
| rastrigin_3d | **PBO+ (full)** | 10 | 2.4e+00 | 2.0e+00 | 4.4e+00 | 736 (1/10) | 747 (1/10) |
| rastrigin_3d | PBO+ w/o critic | 10 | 4.5e+00 | 2.4e+00 | 5.0e+00 | – (0/10) | – (0/10) |
| rastrigin_3d | natural-gradient PBO (xNES) | 10 | 2.2e+00 | 2.0e+00 | 2.7e+00 | 695 (1/10) | – (0/10) |
| rastrigin_3d | CMA-ES (pycma) | 10 | 3.1e+00 | 2.1e+00 | 4.7e+00 | 476 (1/10) | 695 (1/10) |
| rastrigin_3d | lq-CMA-ES (pycma) | 10 | 3.0e+00 | 1.9e+00 | 6.2e+00 | – (0/10) | – (0/10) |
| rosenbrock_20d | **PBO+ (full)** | 10 | 3.3e-28 | 3.0e-28 | 3.6e-28 | 5103 (9/10) | 5340 (9/10) |
| rosenbrock_20d | PBO+ w/o critic | 10 | 5.7e+00 | 4.0e+00 | 6.7e+00 | – (0/10) | – (0/10) |
| rosenbrock_20d | natural-gradient PBO (xNES) | 10 | 1.7e+01 | 1.6e+01 | 1.7e+01 | – (0/10) | – (0/10) |
| rosenbrock_20d | CMA-ES (pycma) | 10 | 2.5e+00 | 1.1e+00 | 3.2e+00 | – (0/10) | – (0/10) |
| rosenbrock_20d | lq-CMA-ES (pycma) | 10 | 3.0e-14 | 2.1e-14 | 3.1e-14 | 5090 (10/10) | 5452 (10/10) |

