| Study | Method | Completed runs | DE individuals | DE generations | DE evaluations | Adam updates | Total evaluations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previous | pinn | 3 | N/A | N/A | 0 | 5,000 | 5,000 |
| previous | pinn-de | 3 | 8 | 20 | 168 | 4,832 | 5,000 |
| followup | pinn | 3 | N/A | N/A | 0 | 8,064 | 8,064 |
| followup | pinn-de | 3 | 32 | 100 | 3,232 | 4,832 | 8,064 |

Metrics are means ± sample standard deviations; run counts are shown above.

| Study | Method | Reused test RMSE | PDE RMSE | Initial RMSE | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| previous | pinn | 0.246557 ± 0.008680 | 0.112181 ± 0.008270 | 0.157559 ± 0.017072 | 252.1 ± 7.8 |
| previous | pinn-de | 0.258089 ± 0.015586 | 0.124687 ± 0.066915 | 0.157038 ± 0.076131 | 245.9 ± 8.4 |
| followup | pinn | 0.317388 ± 0.014733 | 0.030829 ± 0.006168 | 0.308117 ± 0.025918 | 397.0 ± 20.0 |
| followup | pinn-de | 0.255862 ± 0.009800 | 0.098389 ± 0.019697 | 0.169772 ± 0.031687 | 326.6 ± 3.8 |
