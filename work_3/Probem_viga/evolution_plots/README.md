# Evolution plots

Run `../plot_results.py` after generating the independent-run curves and
run-level results. For each experimental configuration, the script selects the
execution with the lowest `best_f` and saves:

- a PNG with the mean population fitness of the best execution; and
- a band representing one population-fitness standard deviation above and
  below that mean.

The title reports the best seed and the final-population mean and standard
deviation of the adaptive `F` and `CR` parameters. These statistics describe
the final population of the best execution, not only its best individual. The
figures do not compare algorithms or models.
