# Figure generation — Paper 1

Scripts that produce Figures 1 to 3 of the CKD sex-disparity staging audit.
All values are hard-coded from the reported results so the figures are
reproducible without access to patient data.

## Requirements

    pip install matplotlib          # Figures 2 and 3
    pip install playwright          # Figure 1 rasterisation only
    playwright install chromium

Figure 1 is written as SVG by hand (no dependencies). Rasterising it to PNG
or PDF needs a renderer; Playwright/Chromium is used here, but any SVG
converter works (Inkscape, rsvg-convert, cairosvg).

## Running

    python fig1_cohort_flow.py      # -> fig/Figure1_cohort_flow.svg
    python fig2_fig3_results.py     # -> fig/Figure2_*.pdf/.png, fig/Figure3_*.pdf/.png

Then rasterise Figure 1, e.g.

    rsvg-convert -f pdf -o fig/Figure1_cohort_flow.pdf fig/Figure1_cohort_flow.svg

## Where the numbers come from

Figure 1   cohort accounting, Methods (2,510 -> 2,472 -> 2,463) and Table 1
Figure 2a  Table 3 (baseline FNR by sex) and Table 8 (Model B)
Figure 2b  Table 5 (all three pipeline phases: FNR, accuracy)
Figure 3   Table 7 (observed proportion, mean predicted, by sex and model)

Changing a reported value means changing it in the script AND the table.
The scripts do not read the analysis output, so they will not catch a
mismatch for you.

## Design constraints

Greyscale-safe: series differ by both fill shade and hatching, so nothing
depends on colour. Zero-height bars are marked with a baseline stub, since
a bar of exactly 0.0% is otherwise invisible and reads as a rendering fault.
Figure 2b uses a deliberately wide right-hand axis (60-90%) so the
1.1-point accuracy change is not visually exaggerated.
