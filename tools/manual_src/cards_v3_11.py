# -*- coding: utf-8 -*-
"""Manual cards for v3.11.0 - rate smoothing and the extended LISA family.

Rendered by tools/build_manual.py.  LaTeX goes in raw strings; a bare '<'
followed by a letter must be written &lt; (the HTML parser would read a tag).
"""

PERM = ("PERMUTATIONS", "Integer", "999", "99–9999",
        "Conditional permutations per unit. The smallest attainable two-sided p-value is 2/(P+1); the report warns when "
        "P is too small for the chosen correction to confirm anything")
SEED = ("RANDOM_SEED", "Integer", "42", "≥ 0", "Seed for reproducible permutations")
MT = ("MULTIPLE_TESTING", "Enum", "Benjamini-Hochberg FDR", "None / FDR / Bonferroni",
      "Correction applied across all tested units before a class is assigned")
INPUT = ("INPUT", "Vector Layer", "—", "Any geometry type", "Input layer; polygons for contiguity weights, any geometry for KNN / distance")
WEIGHTS = [
    ("WEIGHT_TYPE", "Enum", "Queen contiguity", "Queen / Rook / KNN / Distance Band", "Neighbourhood definition"),
    ("KNN", "Integer", "8", "≥ 1", "Number of neighbours (KNN only)"),
    ("DISTANCE_BAND", "Double", "1000.0", "> 0", "Distance threshold in map units (Distance Band only)"),
]
OUT_LAYER = ("OUTPUT", "Vector Layer", "—", "Optional path; auto-styled", "Input features with the result fields appended")
REPORT = ("HTML_REPORT", "File", "temporary", "*.html", "Analyst report")

REF_ANSELIN_1995 = ('<strong>Anselin, L.</strong> (1995). "Local Indicators of Spatial Association — LISA." <em>Geographical '
                    'Analysis</em> 27(2): 93–115. DOI: <a href="https://doi.org/10.1111/j.1538-4632.1995.tb00338.x">'
                    '10.1111/j.1538-4632.1995.tb00338.x</a> — Conditional randomisation for local statistics.')
REF_BH = ('<strong>Benjamini, Y. &amp; Hochberg, Y.</strong> (1995). "Controlling the False Discovery Rate: A Practical and '
          'Powerful Approach to Multiple Testing." <em>Journal of the Royal Statistical Society B</em> 57(1): 289–300. DOI: '
          '<a href="https://doi.org/10.1111/j.2517-6161.1995.tb02031.x">10.1111/j.2517-6161.1995.tb02031.x</a> — The '
          'default multiple-testing correction.')
REF_REY = ('<strong>Rey, S.J., Arribas-Bel, D. &amp; Wolf, L.J.</strong> (2023). <em>Geographic Data Science with Python.</em> '
           'Boca Raton: CRC Press. ISBN: 978-1032445953 — PySAL / esda reference implementations this engine is checked against.')
REF_AR = ('<strong>Assunção, R.M. &amp; Reis, E.A.</strong> (1999). "A new proposal to adjust Moran\'s I for population density." '
          '<em>Statistics in Medicine</em> 18(16): 2147–2162. DOI: <a href="https://doi.org/10.1002/(SICI)1097-0258(19990830)18:16%3C2147::AID-SIM179%3E3.0.CO;2-I">'
          '10.1002/(SICI)1097-0258(19990830)18:16&lt;2147::AID-SIM179&gt;3.0.CO;2-I</a> — The EB-standardised rate.')
REF_MARSHALL = ('<strong>Marshall, R.J.</strong> (1991). "Mapping Disease and Mortality Rates Using Empirical Bayes '
                'Estimators." <em>Journal of the Royal Statistical Society C (Applied Statistics)</em> 40(2): 283–294. DOI: '
                '<a href="https://doi.org/10.2307/2347593">10.2307/2347593</a> — The method-of-moments global and local EB estimators.')
REF_CK = ('<strong>Clayton, D. &amp; Kaldor, J.</strong> (1987). "Empirical Bayes Estimates of Age-Standardized Relative Risks '
          'for Use in Disease Mapping." <em>Biometrics</em> 43(3): 671–681. DOI: <a href="https://doi.org/10.2307/2532003">'
          '10.2307/2532003</a> — Why raw rates on small populations mislead and how shrinkage repairs it.')
REF_ALK = ('<strong>Anselin, L., Lozano-Gracia, N. &amp; Koschinsky, J.</strong> (2006). "Rate Transformations and Smoothing." '
           'Technical Report, Spatial Analysis Laboratory, University of Illinois. — The GeoDa conventions for raw, EB and '
           'spatial rates and the EB Moran\'s I.')
REF_LEE = ('<strong>Lee, S.-I.</strong> (2001). "Developing a bivariate spatial association measure: An integration of '
           'Pearson\'s r and Moran\'s I." <em>Journal of Geographical Systems</em> 3(4): 369–385. DOI: '
           '<a href="https://doi.org/10.1007/s101090100064">10.1007/s101090100064</a> — Defines L, L_XX and the local L_i.')

CARDS = [
    # ------------------------------------------------------------------ 01
    {
        "id": "alg-rate-smoothing", "alg": "rate_smoothing", "after": "alg-export-attrs",
        "title": "Rate Smoothing (Empirical Bayes)", "sidebar": "Rate Smoothing (EB)", "icon": "⚖️",
        "desc": "Turns an event count and a population at risk into rates that can be mapped and tested without small "
                "denominators driving the pattern. Writes the raw rate next to four stabilised alternatives - global and "
                "spatial empirical Bayes, the spatial window rate and the Assunção–Reis EB-standardised rate - so the effect "
                "of each choice is visible on the same map.",
        "background": [
            ("The small-number problem.", "A unit with 40 residents and two incidents has a raw rate of 50 per 1000; a unit "
             "with 40 000 residents needs 2000 incidents for the same rate. Under a Poisson model the variance of a raw rate "
             "is proportional to 1/base, so the most extreme raw rates sit, almost by construction, on the smallest bases. A "
             "choropleth of raw rates therefore draws the eye to the least reliable units, and a LISA treats them as equally "
             "trustworthy. The report's first chart - raw rate against log(base) - shows this as a funnel that narrows to "
             "the right."),
            ("Borrowing strength.", "Empirical Bayes estimators (Clayton &amp; Kaldor 1987; Marshall 1991) shrink each raw "
             "rate towards a prior mean in proportion to how unreliable it is. The global estimator uses the study-area "
             "rate as the prior; the spatial (local) estimator uses the rate of the unit's own neighbourhood, so a small "
             "unit inside a genuinely high-risk area is pulled towards that area's level rather than towards the city "
             "average. Both estimate the prior's variance from the data by the method of moments."),
            ("Standardising instead of smoothing.", "For testing spatial autocorrelation the relevant transform is "
             "Assunção &amp; Reis (1999): each rate's departure from the overall rate divided by its own estimated "
             "standard error. The result is a z-like score with roughly constant variance, which is what Moran's I "
             "assumes; GeoDa's and esda's 'EB rate' Moran statistics use exactly this variable."),
        ],
        "notation": r"Let \(e_i\) be the event count and \(P_i\) the population at risk of unit <em>i</em>, \(r_i = e_i / P_i\) "
                    r"the raw rate, and \(b = \sum e_i / \sum P_i\) the overall rate.",
        "formulas": [
            ("Method-of-moments prior variance:", r"a = \frac{\sum_i P_i (r_i - b)^2}{\sum_i P_i} - \frac{b}{\bar{P}}"),
            ("Global EB rate (Marshall 1991):", r"\hat{\theta}_i = w_i r_i + (1 - w_i)\, b, \qquad w_i = \frac{a}{a + b / P_i}",
             "When \\(a \\le 0\\) - the rates vary less than Poisson noise would make them - \\(a\\) is set to 0 and every "
             "unit takes the overall rate. The report says when this happened."),
            ("Spatial window rate:", r"r_i^{W} = \frac{e_i + \sum_{j \in N(i)} e_j}{P_i + \sum_{j \in N(i)} P_j}"),
            ("Spatial EB rate:", r"\hat{\theta}_i^{S} = r_i^{W} + (r_i - r_i^{W}) \frac{a_i}{a_i + r_i^{W}/P_i}",
             "with \\(a_i\\) the moment variance computed over the unit's window, set to 0 when negative."),
            ("Assunção–Reis EB-standardised rate:", r"z_i = \frac{r_i - b}{\sqrt{a + b / P_i}}",
             "with the variance falling back to \\(b / P_i\\) where \\(a + b/P_i\\) is negative."),
        ],
        "note": "<code>core/local_stats_engines.py</code>: <code>global_eb_rate</code>, <code>spatial_rate</code>, "
                "<code>spatial_eb_rate</code>, <code>assuncao_reis_rate</code>. All four agree with esda's "
                "<code>Empirical_Bayes</code>, <code>Spatial_Rate</code>, <code>Spatial_Empirical_Bayes</code> and "
                "<code>assuncao_rate</code> to floating-point precision (<code>tests/esda_parity.py</code>). One deliberate "
                "difference: esda's global estimator keeps a negative prior variance, which yields negative shrinkage "
                "weights; this tool clamps it to zero, as Marshall and GeoDa do.",
        "params": [INPUT,
                   ("EVENTS", "Numeric Field", "—", "≥ 0", "Event count (numerator)"),
                   ("BASE", "Numeric Field", "—", "> 0", "Population at risk (denominator); units with a zero or negative base are left out"),
                   ("MAP_RATE", "Enum", "Spatial EB", "5 options", "Which rate the output layer is styled by"),
                   ("MULTIPLIER", "Double", "1000", "> 0", "Rates are multiplied by this (1000 = per thousand); the standardised rate is not"),
                   ] + WEIGHTS + [OUT_LAYER, REPORT],
        "outputs": [
            ("rate_raw", "Double", "Raw rate × multiplier"),
            ("rate_geb", "Double", "Global empirical Bayes rate × multiplier"),
            ("rate_sw", "Double", "Spatial window rate × multiplier"),
            ("rate_seb", "Double", "Spatial empirical Bayes rate × multiplier"),
            ("rate_arz", "Double", "Assunção–Reis EB-standardised rate (unitless)"),
            ("eb_weight", "Double", "Weight on the unit's own raw rate in the global EB estimate (1 = kept, 0 = fully shrunk)"),
            ("seb_weight", "Double", "Same, for the spatial EB estimate"),
        ],
        "report": "KPI row (units, study-area rate, raw-rate spread on the smallest vs the largest bases, mean shrinkage "
                  "weight); a summary table of all five rates; two scatter plots of raw and EB rate against log(base); "
                  "the ten highest raw rates with their shrinkage and their rank after smoothing. The output layer gets a "
                  "quantile renderer on the selected rate.",
        "interp_title": "Reading the five rates.",
        "interp": [
            ("Raw-rate spread on small bases far above that on large bases",
             "Classic small-number instability", "Map and test a smoothed or standardised rate, not the raw one."),
            ("eb_weight close to 1 everywhere", "Bases are large enough that smoothing changes little",
             "Raw rates can be reported; smoothing remains a safe default."),
            ("A unit's rank moves a lot between rate_raw and rate_geb", "Its raw rate rests on a small base",
             "Do not single it out from the raw value; report the EB estimate with the raw figure beside it."),
            ("Negative prior variance reported", "No between-area variation beyond Poisson noise at this scale",
             "There is no rate pattern to map; consider a coarser unit or a longer period."),
        ],
        "see_also": ["alg-eb-rate-lisa", "alg-global-moran", "alg-getis-ord"],
        "refs": [REF_MARSHALL, REF_CK, REF_AR, REF_ALK, REF_REY],
    },
    # ------------------------------------------------------------------ 02
    {
        "id": "alg-gamma", "alg": "gamma_index", "after": "alg-geodetector-q",
        "title": "Gamma Index of Spatial Association", "sidebar": "Gamma Index", "icon": "Γ",
        "desc": "The general cross-product statistic: the sum, over every pair of neighbours, of a chosen attribute-"
                "similarity score. Moran's I, Geary's C and the join counts are special cases; Gamma lets the analyst "
                "choose the similarity measure and reports all three standard choices side by side, so a conclusion that "
                "depends on how similarity is defined becomes visible.",
        "background": [
            ("A family, not a statistic.", "Hubert, Golledge &amp; Costanzo (1981) showed that most measures of spatial "
             "autocorrelation can be written as \\(\\Gamma = \\sum_i \\sum_j w_{ij} a_{ij}\\): a spatial-proximity matrix "
             "matched against an attribute-similarity matrix. With \\(a_{ij}\\) a cross-product of deviations and "
             "row-standardised weights, Gamma is Moran's I up to scaling; with squared differences it is Geary's C."),
            ("Why choose the measure.", "The cross-product rewards pairs that are both far from the mean in the same "
             "direction; squared differences reward pairs that are close to each other wherever they lie; absolute "
             "differences do the same with less weight on a few extreme pairs. On skewed urban data the three can "
             "disagree, and a pattern that appears under only one of them is a pattern in the choice of measure."),
            ("Inference.", "Total randomisation: the values are shuffled over the locations, Gamma is recomputed, and "
             "the observed value is placed in that reference distribution. No distributional assumption is needed."),
        ],
        "formulas": [
            ("General cross-product:", r"\Gamma = \sum_{i}\sum_{j} w_{ij}\, a_{ij}"),
            ("The three similarity measures:", r"a_{ij}^{c} = y_i y_j, \qquad a_{ij}^{s} = (y_i - y_j)^2, \qquad a_{ij}^{a} = |y_i - y_j|",
             "Weights are binary (\\(w_{ij} = 1\\) for neighbours). With the cross-product, standardising y first is "
             "strongly advised - otherwise Gamma is dominated by the squared mean."),
            ("Permutation z and p:", r"z = \frac{\Gamma - \overline{\Gamma^{(p)}}}{\mathrm{sd}(\Gamma^{(p)})}, \qquad "
             r"p = \min\!\left(1,\; \frac{2\,(\min(\#\{\Gamma^{(p)} \ge \Gamma\},\, \#\{\Gamma^{(p)} \le \Gamma\}) + 1)}{P + 1}\right)"),
        ],
        "note": "Positive autocorrelation shows as a <em>large</em> Gamma for the cross-product and as a <em>small</em> one "
                "for the two difference measures; the report states the direction in words. The statistic equals esda's "
                "<code>Gamma</code> for all three operations (<code>tests/esda_parity.py</code>).",
        "params": [INPUT,
                   ("FIELD", "Numeric Field", "—", "Must vary", "Variable to test"),
                   ("OPERATION", "Enum", "Cross-product", "Cross-product / squared / absolute difference", "Measure shown first in the report (all three are computed)"),
                   ("STANDARDIZE", "Boolean", "True", "—", "z-score the field before computing Gamma"),
                   ] + WEIGHTS + [("PERMUTATIONS", "Integer", "999", "99–99999", "Random permutations"), SEED, REPORT],
        "outputs": [],
        "report": "KPI row (Gamma, permutation mean, z, p); a verdict sentence; a histogram of the reference distribution "
                  "with the observed value marked; a table of all three similarity measures with their readings, and a "
                  "warning when they disagree.",
        "interp_title": "Reading Gamma.",
        "interp": [
            ("All three measures significant in the positive direction", "Robust positive spatial association",
             "Proceed to local statistics (LISA, Gi*)."),
            ("Only the cross-product is significant", "Driven by a few units far from the mean",
             "Inspect the extremes; consider a transformation of the field."),
            ("Measures disagree in direction", "The finding depends on the definition of similarity",
             "Report it as fragile; check the extreme pairs before drawing conclusions."),
        ],
        "see_also": ["alg-global-moran", "alg-geary-c", "alg-join-count"],
        "refs": [
            '<strong>Hubert, L.J., Golledge, R.G. &amp; Costanzo, C.M.</strong> (1981). "Generalized Procedures for '
            'Evaluating Spatial Autocorrelation." <em>Geographical Analysis</em> 13(3): 224–233. DOI: '
            '<a href="https://doi.org/10.1111/j.1538-4632.1981.tb00731.x">10.1111/j.1538-4632.1981.tb00731.x</a> — The '
            'general cross-product framework.',
            '<strong>Cliff, A.D. &amp; Ord, J.K.</strong> (1981). <em>Spatial Processes: Models &amp; Applications.</em> '
            'London: Pion. ISBN: 978-0850860818 — Moran, Geary and join counts as cross-product statistics.',
            REF_REY,
        ],
    },
    {
        "id": "alg-smaup", "alg": "smaup_test", "after": "alg-gamma",
        "title": "S-maup: Is the MAUP Effect Significant?", "sidebar": "S-maup (MAUP test)", "icon": "🧩",
        "desc": "Before aggregating n areas into k larger zones, tests whether the distribution of the aggregated "
                "variable will depend on how the zones are drawn - the modifiable areal unit problem - using the "
                "S-maup statistic of Duque, Laniado &amp; Polo (2018). Reports the verdict for the planned k and the "
                "number of zones beyond which aggregation is safe.",
        "background": [
            ("The modifiable areal unit problem.", "Openshaw &amp; Taylor (1979) showed that the same data aggregated "
             "into different zones can give correlations anywhere from strongly negative to strongly positive. Any "
             "statistic of an aggregated variable carries the zoning as a hidden parameter. The question for a planner is "
             "not whether the MAUP exists but whether it matters for this variable at this level of aggregation."),
            ("Three numbers decide it.", "Duque et al. ran a large simulation over grids of 25 to 900 areas, aggregating "
             "each into k zones in many different ways, and measured how much the distribution of the aggregated "
             "variable moved. The sensitivity depends on n, k and the spatial autocorrelation rho of the variable: "
             "strong positive autocorrelation makes aggregation safe (neighbours are alike, so any contiguous grouping of "
             "them gives similar zone values), weak or negative autocorrelation makes it fragile, and few large zones "
             "are more fragile than many small ones."),
            ("A fitted statistic with tabulated critical values.", "The paper fits a closed-form curve to the "
             "simulated sensitivity and tabulates critical values by rho and n. This tool measures rho as the variable's "
             "Global Moran's I, evaluates the curve and compares it with the critical value of the nearest tabulated rho, "
             "interpolated in n."),
        ],
        "formulas": [
            ("Aggregation ratio:", r"\theta = k / n"),
            ("S-maup (Duque et al. 2018):", r"S = \frac{L}{1 + \eta\, e^{\rho\,\tau}}, \qquad L = \frac{1}{1 + e^{-2.2 + 7.03\,\theta}}, "
             r"\quad \eta = e^{-0.6618}\, \theta^{1.3}, \quad \tau = 5.32 - 5.53\,\theta"),
            ("Decision:", r"\text{MAUP effect significant at level } \alpha \iff S > c_{\alpha}(\rho, n)"),
        ],
        "note": "The statistic and the critical values match esda's <code>Smaup</code> exactly. One deliberate "
                "difference: esda selects the table row through open intervals, so a rho of exactly 0.8, 0.6, 0.4, 0.15 "
                "(or any rho of 1) falls through to the -0.9 row; this tool uses the nearest row, which has no gap. n "
                "outside 25–900 is clamped and flagged as an extrapolation, as in the paper.",
        "params": [INPUT,
                   ("FIELD", "Numeric Field", "—", "Must vary", "Variable to be aggregated"),
                   ("K_REGIONS", "Integer", "10", "1 ≤ k ≤ n", "Number of zones you plan to aggregate into"),
                   ] + WEIGHTS + [("PERMUTATIONS", "Integer", "999", "99–9999", "Permutations for Global Moran's I"), SEED, REPORT],
        "outputs": [],
        "report": "KPI row (S-maup, 5% critical value, Moran's I, verdict); the critical-value table at 10/5/1%; the "
                  "largest number of zones at which the effect is still significant; a curve of S-maup and the 5% "
                  "critical value against k with the planned k marked.",
        "interp_title": "Reading the S-maup verdict.",
        "interp": [
            ("Significant at the planned k", "Zone statistics will depend on the zoning",
             "Report results for several zonings, or build zones from the data (Max-p, SKATER)."),
            ("Not significant at the planned k", "Aggregation into k zones is robust for this variable",
             "Proceed with the planned zoning."),
            ("Significant only for small k", "Few large zones are fragile, many small ones are safe",
             "Choose at least the reported number of zones."),
        ],
        "see_also": ["alg-global-moran", "alg-skater", "alg-sensitivity"],
        "refs": [
            '<strong>Duque, J.C., Laniado, H. &amp; Polo, A.</strong> (2018). "S-maup: Statistical test to measure the '
            'sensitivity to the modifiable areal unit problem." <em>PLOS ONE</em> 13(11): e0207377. DOI: '
            '<a href="https://doi.org/10.1371/journal.pone.0207377">10.1371/journal.pone.0207377</a> — The statistic and '
            'its critical-value table.',
            '<strong>Openshaw, S. &amp; Taylor, P.J.</strong> (1979). "A Million or So Correlation Coefficients: Three '
            'Experiments on the Modifiable Areal Unit Problem." In N. Wrigley (ed.), <em>Statistical Applications in the '
            'Spatial Sciences</em>, pp. 127–144. London: Pion. — The demonstration that zoning changes results.',
            '<strong>Openshaw, S.</strong> (1984). <em>The Modifiable Areal Unit Problem.</em> CATMOG 38. Norwich: Geo Books. '
            '— The classic monograph.',
        ],
    },
    {
        "id": "alg-global-lee-l", "alg": "global_bivariate_lee_l", "after": "alg-join-count",
        "title": "Global Bivariate Lee's L", "sidebar": "Global Bivariate Lee's L", "icon": "🔗",
        "desc": "Lee's (2001) L for two fields over the whole study area - also called the spatial Pearson coefficient. "
                "Both fields are z-scored and spatially smoothed, and L is the average product of the smoothed values, "
                "so it combines the plain correlation of the two fields with how strongly each is spatially clustered. "
                "<strong>Corrected in 3.11.0:</strong> earlier versions computed the global bivariate Moran statistic "
                "under this name; values from 3.10 and before are not Lee's L.",
        "background": [
            ("Why Lee proposed L.", "Pearson's r ignores where observations are; bivariate Moran's I (Wartenberg 1985) "
             "relates x at a location to the lag of y around it, which makes it asymmetric - swapping x and y changes "
             "the answer - and lets it be driven by the spatial autocorrelation of y alone. Lee (2001) smooths both "
             "variables with the same spatial filter and correlates the smoothed fields, giving a symmetric statistic "
             "that reaches its extremes only when the two fields are correlated <em>and</em> both are spatially "
             "clustered."),
            ("The decomposition.", "Lee shows that L factors into the spatial smoothing scalars of the two fields and the "
             "correlation of their smoothed versions, \\(L = \\sqrt{SSS_X}\\sqrt{SSS_Y}\\, r_{\\tilde{X}\\tilde{Y}}\\). "
             "The report prints \\(L_{XX}\\) and \\(L_{YY}\\) (the SSS of each field) next to the raw and the smoothed "
             "correlation so the reader can see which factor drives L."),
            ("What was wrong before 3.11.0.", "The tool previously averaged \\(z_{x,i}\\) times the spatial lag of "
             "\\(z_y\\) - the bivariate Moran statistic, which Bivariate LISA already reports. It was not symmetric and "
             "did not include the unit's own value in the smoothing. The correction was found by comparing the plugin "
             "with esda's <code>Spatial_Pearson</code> while building v3.11.0."),
        ],
        "notation": r"Let \(z_x, z_y\) be the z-scored fields and \(V\) the smoothing matrix: binary contiguity, optionally "
                    r"with the unit itself included (Lee's recommendation), then row-standardised.",
        "formulas": [
            ("Smoothed fields:", r"\tilde{x} = V z_x, \qquad \tilde{y} = V z_y"),
            ("Lee's L:", r"L_{XY} = \frac{\sum_i \tilde{x}_i\, \tilde{y}_i}{\sum_i \left(\sum_j v_{ij}\right)^2}"),
            ("Spatial smoothing scalars:", r"L_{XX} = \frac{\sum_i \tilde{x}_i^2}{\sum_i (\sum_j v_{ij})^2}, \qquad L_{XY} \approx \sqrt{L_{XX}}\sqrt{L_{YY}}\; r(\tilde{x}, \tilde{y})"),
            ("Two permutation tests:", r"p_{\text{pairs}}:\ (x_i, y_i) \text{ shuffled together}, \qquad p_{\text{any}}:\ y \text{ shuffled against fixed } x",
             "The pairs test keeps the plain correlation and asks whether the spatial co-pattern goes beyond it (esda's "
             "convention); the second asks whether there is any bivariate spatial association at all."),
        ],
        "note": "<code>core/local_stats_engines.py::lee_l</code>. Without the self-window option L equals esda's "
                "<code>Spatial_Pearson</code> association to floating-point precision. With it, the unit enters its own "
                "window with the same weight as each neighbour (1/(k+1)) - not with half the window, which is what adding "
                "a self-weight of 1 to already row-standardised weights would give; a unit test pins this.",
        "params": [INPUT,
                   ("FIELD_X", "Numeric Field", "—", "Must vary", "First field"),
                   ("FIELD_Y", "Numeric Field", "—", "Must vary; ≠ X", "Second field"),
                   ("WEIGHT_TYPE", "Enum", "Queen contiguity", "Queen / Rook / KNN / Distance Band", "Neighbourhood definition (used as binary)"),
                   ("KNN", "Integer", "5", "≥ 1", "Neighbours (KNN only)"),
                   ("DISTANCE_BAND", "Double", "1000.0", "> 0", "Distance band (Distance Band only)"),
                   ("INCLUDE_SELF", "Boolean", "True", "—", "Include each unit in its own smoothing window"),
                   ("PERMUTATIONS", "Integer", "999", "99–99999", "Permutations for both tests"), SEED, REPORT],
        "outputs": [],
        "report": "KPI row (L with z, both p-values, Pearson r); a verdict; the decomposition table (L, L_XX, L_YY, raw and "
                  "smoothed r, permutation mean and sd); a scatter of smoothed X against smoothed Y.",
        "interp_title": "Reading Lee's L.",
        "interp": [
            ("Both p-values significant", "Spatial co-clustering beyond the plain correlation",
             "Map it with Bivariate Spatial Association (Lee's L)."),
            ("Only the Y-shuffled p significant", "The association is the plain correlation; no extra spatial co-pattern",
             "Report the correlation; a spatial story is not supported."),
            ("Small L_XX or L_YY", "One field is barely clustered, which caps L",
             "Check that field with Global Moran's I."),
        ],
        "see_also": ["alg-bivariate-lee", "alg-bivariate-lisa", "alg-global-moran"],
        "refs": [REF_LEE,
                 '<strong>Wartenberg, D.</strong> (1985). "Multivariate Spatial Correlation: A Method for Exploratory '
                 'Geographical Analysis." <em>Geographical Analysis</em> 17(4): 263–283. DOI: '
                 '<a href="https://doi.org/10.1111/j.1538-4632.1985.tb00849.x">10.1111/j.1538-4632.1985.tb00849.x</a> — '
                 'The bivariate Moran statistic Lee\'s L improves on.', REF_REY],
    },
    # ------------------------------------------------------------------ 03
    {
        "id": "alg-bivariate-lee", "alg": "bivariate_spatial_association_lees_l", "after": "alg-bivariate-lisa",
        "title": "Bivariate Spatial Association (Lee's L)", "sidebar": "Bivariate Lee's L", "icon": "🔗",
        "desc": "Local Lee's L: for each unit, the product of the spatially smoothed z-scores of two fields, with a "
                "conditional permutation test and multiple-testing control. HH and LL units are neighbourhoods that are "
                "high (or low) on both fields together. <strong>Corrected in 3.11.0:</strong> earlier versions computed "
                "the bivariate local Moran statistic under this name and had no significance test.",
        "background": [
            ("Local L.", "Lee (2001) decomposes the global L into one term per unit, \\(L_i = \\tilde{x}_i \\tilde{y}_i\\) "
             "under row-standardised smoothing: the product of the neighbourhood averages of the two standardised fields. "
             "Because both fields are smoothed, the map is the same whichever field is called X."),
            ("Lee's L or Bivariate LISA?", "Bivariate LISA asks an asymmetric question - is x here related to y around "
             "here? - and is the right tool for spillover hypotheses. Local Lee's L asks whether the neighbourhood as a "
             "whole is high or low on both, which is the planning question behind 'where do density and accessibility "
             "coincide?'."),
            ("Inference.", "Unit i is held fixed in its own window and its neighbours are redrawn from the other n-1 units "
             "(conditional randomisation, Anselin 1995); both smoothed values are recomputed from the same draw. Classes "
             "are assigned only where the corrected p-value is below 0.05."),
        ],
        "formulas": [
            ("Local Lee's L:", r"L_i = \tilde{x}_i\, \tilde{y}_i = \Big(\sum_j v_{ij} z_{x,j}\Big)\Big(\sum_j v_{ij} z_{y,j}\Big)"),
            ("Class (significant units only):", r"\text{HH: } \tilde{x}_i \ge 0, \tilde{y}_i \ge 0; \quad \text{LL: both } \lt 0; \quad \text{HL / LH: signs differ}"),
        ],
        "note": "Equal to esda's <code>Spatial_Pearson_Local</code> association without the self-window. The output field "
                "<code>y_lag_z</code> of earlier versions is replaced by <code>x_smooth</code> and <code>y_smooth</code>.",
        "params": [INPUT,
                   ("X_FIELD", "Numeric Field", "—", "Must vary", "First field"),
                   ("Y_FIELD", "Numeric Field", "—", "Must vary; ≠ X", "Second field"),
                   ("WEIGHT_TYPE", "Enum", "KNN", "Queen / Rook / KNN / Distance Band", "Neighbourhood definition (used as binary)"),
                   ("KNN", "Integer", "5", "≥ 1", "Neighbours (KNN only)"),
                   ("DISTANCE_BAND", "Double", "1000.0", "> 0", "Distance band (Distance Band only)"),
                   ("INCLUDE_SELF", "Boolean", "True", "—", "Include each unit in its own smoothing window"),
                   PERM, SEED, MT, OUT_LAYER, REPORT],
        "outputs": [
            ("lee_l", "Double", "Local Lee's L"),
            ("x_smooth", "Double", "Smoothed z-score of X"),
            ("y_smooth", "Double", "Smoothed z-score of Y"),
            ("lee_p", "Double", "Two-sided conditional-permutation p-value"),
            ("lee_padj", "Double", "Corrected p-value"),
            ("lee_class", "String", "HH / LL / HL / LH / Not Significant"),
            ("lee_nbrs", "Integer", "Neighbours used"),
        ],
        "report": "KPI row (global L and p, raw and smoothed r, significant units); donut of classes in the LISA colours; "
                  "scatter of the smoothed fields. Output styled with the shared LISA quadrant renderer.",
        "interp_title": "Reading local Lee's L.",
        "interp": [
            ("HH cluster", "Neighbourhood high on both fields", "Candidate area where both conditions coincide."),
            ("HL / LH", "Neighbourhood high on one, low on the other", "A local trade-off or mismatch worth checking on the ground."),
            ("No significant units but a significant global L", "Association spread thinly everywhere",
             "Report the global result; there is no concentrated co-cluster."),
        ],
        "see_also": ["alg-global-lee-l", "alg-bivariate-lisa", "alg-local-moran"],
        "refs": [REF_LEE, REF_ANSELIN_1995, REF_BH],
    },
    {
        "id": "alg-eb-rate-lisa", "alg": "eb_rate_lisa", "after": "alg-skater",
        "title": "Rate Cluster and Outlier Analysis (EB Local Moran)", "sidebar": "EB Rate LISA", "icon": "🩺",
        "desc": "Local Moran's I for rates. Events and population at risk are converted into the Assunção–Reis "
                "EB-standardised rate before the LISA is computed, so clusters cannot be manufactured by the variance "
                "of rates on small bases. With the comparison option on, the same test is run on the raw rate and the "
                "two classifications are cross-tabulated, naming the clusters that exist only because of small numbers.",
        "background": [
            ("Why raw-rate LISAs mislead.", "Local Moran's I assumes every value has the same variance. The variance of a "
             "raw rate is proportional to 1/population, so a tiny unit swings to an extreme value by chance and pulls "
             "its neighbourhood into an HH or LL class - or produces a spurious HL/LH outlier. Assunção &amp; Reis (1999) "
             "showed that this inflates the false-positive rate of Moran's I on rates and proposed the EB "
             "standardisation used here; Anselin et al. (2006) adopted it as GeoDa's 'EB rate' LISA."),
            ("What the cross-table shows.", "Rows are the raw-rate class, columns the EB-rate class. Units significant "
             "on the raw rate but not on the EB rate are small-number artefacts; units that become significant only "
             "after stabilisation are typically large-base units whose genuine difference was masked by noisy small "
             "neighbours."),
        ],
        "formulas": [
            ("EB-standardised rate:", r"z_i = \frac{r_i - b}{\sqrt{a + b / P_i}}"),
            ("Local Moran's I (esda convention):", r"I_i = (n - 1)\, \frac{z^{c}_i \sum_j w_{ij} z^{c}_j}{\sum_k (z^{c}_k)^2}, \qquad z^{c} = z - \bar{z}"),
            ("Conditional permutation, two-sided:", r"p_i = \min\!\left(1,\ \frac{2\,(\min(\#\{I_i^{(p)} \ge I_i\},\, \#\{I_i^{(p)} \le I_i\}) + 1)}{P + 1}\right)"),
        ],
        "note": "The local statistic equals esda's <code>Moran_Local_Rate</code> (which applies the same "
                "<code>assuncao_rate</code> transform) to floating-point precision; p-values agree in rank order "
                "(different random streams). The global EB Moran's I is reported beside the raw-rate Moran's I.",
        "params": [INPUT,
                   ("EVENTS", "Numeric Field", "—", "≥ 0", "Event count"),
                   ("BASE", "Numeric Field", "—", "> 0", "Population at risk"),
                   ] + WEIGHTS + [PERM, SEED, MT,
                                  ("COMPARE_RAW", "Boolean", "True", "—", "Also run the LISA on the raw rate and cross-tabulate"),
                                  OUT_LAYER, REPORT],
        "outputs": [
            ("rate_raw", "Double", "Raw rate"),
            ("eb_z", "Double", "Assunção–Reis EB-standardised rate"),
            ("lisa_i", "Double", "Local Moran's I of eb_z"),
            ("lisa_p", "Double", "Two-sided conditional-permutation p-value"),
            ("lisa_padj", "Double", "Corrected p-value used for the class"),
            ("quadrant", "String", "HH / LL / HL / LH / Not Significant"),
            ("raw_quad", "String", "Class the raw rate would have produced (when compared)"),
            ("lisa_nbrs", "Integer", "Neighbours used"),
        ],
        "report": "KPI row (EB and raw Moran's I with p, cluster and outlier counts); donut of classes; raw-versus-EB "
                  "cross-table with the count of small-number artefacts; a permutation-resolution warning when needed.",
        "interp_title": "Reading the EB rate LISA.",
        "interp": [
            ("HH/LL on the EB rate", "A contiguous area of genuinely high or low risk", "Target for intervention or further study."),
            ("Significant on the raw rate only", "Likely small-number artefact", "Do not act on it; monitor or pool more years."),
            ("Significant on the EB rate only", "A real difference previously hidden by noisy neighbours", "Treat as a finding."),
        ],
        "see_also": ["alg-rate-smoothing", "alg-local-moran", "alg-getis-ord"],
        "refs": [REF_AR, REF_ALK, REF_ANSELIN_1995, REF_BH],
    },
    {
        "id": "alg-mv-local-geary", "alg": "multivariate_local_geary", "after": "alg-eb-rate-lisa",
        "title": "Multivariate Local Geary (Cluster Analysis on Several Fields)", "sidebar": "Multivariate Local Geary", "icon": "🧬",
        "desc": "Anselin's (2019) multivariate Local Geary: is a unit similar to its neighbours across several attributes "
                "at once? Finds places whose overall profile matches their surroundings even when no single field is "
                "extreme, and places that break from their surroundings on the combination of fields.",
        "background": [
            ("From one variable to several.", "The univariate Local Geary sums weighted squared differences between a "
             "unit and its neighbours. Anselin (2019) averages that sum over k z-scored variables, which is the "
             "weighted squared Euclidean distance between the unit and its neighbours in attribute space."),
            ("Positive and negative association.", "A significantly small statistic is positive local association - a "
             "multivariate cluster; a significantly large one is negative association - a multivariate local outlier. "
             "Unlike the univariate case there are no HH/LL labels: the report profiles the clusters field by field so "
             "the analyst can see what they share."),
            ("A strict cut-off.", "Anselin recommends a strict significance threshold because the test is repeated for "
             "every unit and the reference distributions are correlated; the FDR correction is on by default."),
        ],
        "formulas": [
            ("Multivariate Local Geary:", r"c_i = \frac{1}{k} \sum_{v=1}^{k} \sum_j w_{ij}\, (z_{v,i} - z_{v,j})^2"),
            ("Classification (significant units):", r"\text{Positive if } c_i \lt \overline{c_i^{(p)}}, \qquad \text{Negative if } c_i \gt \overline{c_i^{(p)}}"),
        ],
        "note": "Variables are z-scored with the population standard deviation and weights are row-standardised, exactly "
                "as in esda's <code>Geary_Local_MV</code>; the statistic matches it to floating-point precision. The "
                "report warns when two fields correlate above 0.8, because a duplicated dimension counts twice.",
        "params": [INPUT, ("FIELDS", "Numeric Fields", "—", "Two or more, each varying", "Variables to combine")]
                  + WEIGHTS + [PERM, SEED, MT, OUT_LAYER, REPORT],
        "outputs": [
            ("mlg_c", "Double", "Multivariate local Geary statistic"),
            ("mlg_p", "Double", "Two-sided conditional-permutation p-value"),
            ("mlg_padj", "Double", "Corrected p-value"),
            ("mlg_class", "String", "Positive association / Negative association / Not Significant / No neighbours"),
            ("mlg_nbrs", "Integer", "Neighbours used"),
        ],
        "report": "KPI row; class donut; the mean z-score of every field among positive-association and negative-association "
                  "units, as a table and a bar chart; a warning for highly correlated fields.",
        "interp_title": "Reading the multivariate Local Geary.",
        "interp": [
            ("Positive association", "Unit and neighbours share a profile", "Read the profile table; candidate homogeneous zone."),
            ("Negative association", "Unit differs from its neighbours on the combination", "Check what makes it stand out."),
            ("Few significant units", "Profiles vary smoothly or the field set is too diffuse", "Narrow the set of fields."),
        ],
        "see_also": ["alg-local-geary-c", "alg-multivar-cluster", "alg-skater"],
        "refs": [
            '<strong>Anselin, L.</strong> (2019). "A Local Indicator of Multivariate Spatial Association: Extending '
            'Geary\'s c." <em>Geographical Analysis</em> 51(2): 133–150. DOI: <a href="https://doi.org/10.1111/gean.12164">'
            '10.1111/gean.12164</a> — The statistic implemented here.',
            REF_ANSELIN_1995, REF_BH],
    },
    {
        "id": "alg-local-join-counts", "alg": "local_join_counts", "after": "alg-mv-local-geary",
        "title": "Local Join Count Statistics (Binary Clusters and Co-location)", "sidebar": "Local Join Counts", "icon": "🔲",
        "desc": "Local join counts (Anselin &amp; Li 2019) for presence/absence data: where do units with a property cluster, "
                "where do two properties coincide, and where do two properties sit next to each other without "
                "overlapping? The global Join Count Statistics tool says whether 1s cluster somewhere; this tool says "
                "where.",
        "background": [
            ("From the global to the local join count.", "The BB join count of Cliff &amp; Ord counts neighbouring pairs "
             "that are both 1. Its local version counts, for each unit that is 1, how many of its neighbours are 1. "
             "Anselin &amp; Li (2019) made it operational with a one-sided conditional permutation test: the focal unit is "
             "held fixed and its neighbours are drawn at random from the other units."),
            ("Bivariate and multivariate forms.", "Co-location (CLC) counts neighbours where both A and B are 1 around a "
             "unit where both are 1. The bivariate join (BJC) is for two mutually exclusive conditions: a unit with A but "
             "not B, surrounded by units with B but not A. The multivariate form requires every selected variable to be 1."),
            ("Eligibility.", "The statistic is defined only where the focal unit meets its condition. Elsewhere it is 0 "
             "with a NULL p-value and the class 'Not eligible' - not evidence of absence."),
        ],
        "formulas": [
            ("Univariate local join count:", r"BB_i = x_i \sum_j w_{ij}\, x_j"),
            ("Co-location:", r"CLC_i = x_i z_i \sum_j w_{ij}\, x_j z_j"),
            ("Bivariate join:", r"BJC_i = x_i (1 - z_i) \sum_j w_{ij}\, (1 - x_j)\, z_j"),
            ("One-sided permutation p:", r"p_i = \frac{\#\{BB_i^{(p)} \ge BB_i\} + 1}{P + 1}"),
        ],
        "note": "Binary weights. The counts equal esda's <code>Join_Counts_Local</code>, <code>_BV</code> (CLC and BJC) and "
                "<code>_MV</code>. One deliberate difference: esda's BJC simulation drops the x<sub>j</sub> = 0 condition; "
                "here the simulated statistic uses exactly the observed definition, with each (x, z) pair permuted together.",
        "params": [INPUT,
                   ("FIELD_A", "Numeric Field", "—", "0/1 or recodable", "First variable"),
                   ("MODE", "Enum", "Univariate", "Univariate / CLC / BJC / Multivariate", "Join type"),
                   ("FIELD_B", "Numeric Field", "—", "Required for CLC and BJC", "Second variable"),
                   ("EXTRA_FIELDS", "Numeric Fields", "—", "Multivariate mode", "Further variables"),
                   ("RULE", "Enum", "Fields are already 0/1", "0/1 / above median / at or above threshold", "How values become 0/1"),
                   ("THRESHOLD", "Double", "1.0", "—", "Threshold for the threshold rule"),
                   ] + WEIGHTS + [PERM, SEED, MT, OUT_LAYER, REPORT],
        "outputs": [
            ("ljc", "Double", "Local join count"),
            ("ljc_p", "Double", "One-sided permutation p-value (NULL where not tested)"),
            ("ljc_padj", "Double", "Corrected p-value"),
            ("ljc_class", "String", "Significant cluster / Not Significant / Not eligible"),
            ("ljc_nbrs", "Integer", "Neighbours"),
        ],
        "report": "KPI row (eligible units, significant clusters, mean joins); class donut; the share of units coded 1 per "
                  "field; a warning when fewer than ten units are eligible.",
        "interp_title": "Reading local join counts.",
        "interp": [
            ("Significant cluster", "More neighbours share the property than chance allows", "Delineate the contiguous area."),
            ("Eligible but not significant", "The unit has the property but its neighbours mostly do not", "An isolated case."),
            ("Not eligible", "The focal condition is not met", "No statement is made about this unit."),
        ],
        "see_also": ["alg-join-count", "alg-colocation-quotient", "alg-local-moran"],
        "refs": [
            '<strong>Anselin, L. &amp; Li, X.</strong> (2019). "Operational local join count statistics for cluster '
            'detection." <em>Journal of Geographical Systems</em> 21(2): 189–210. DOI: '
            '<a href="https://doi.org/10.1007/s10109-019-00299-x">10.1007/s10109-019-00299-x</a> — The statistics and '
            'their permutation test.',
            '<strong>Cliff, A.D. &amp; Ord, J.K.</strong> (1981). <em>Spatial Processes: Models &amp; Applications.</em> '
            'London: Pion. ISBN: 978-0850860818 — The global join count statistics.', REF_BH],
    },
    {
        "id": "alg-losh", "alg": "losh_heteroskedasticity", "after": "alg-local-join-counts",
        "title": "Local Spatial Heteroskedasticity (LOSH)", "sidebar": "LOSH", "icon": "📶",
        "desc": "Ord &amp; Getis's (2012) H<sub>i</sub>: how variable a field is inside each unit's neighbourhood - the "
                "counterpart of Gi*, which measures how high it is. Combined with Gi* it separates uniform hot spots from "
                "hot spots made of mixed values and marks transition zones between regimes.",
        "background": [
            ("Level and variability.", "A Gi* hot spot says the neighbourhood average is high; it does not say whether "
             "every unit is high or a few very high units sit among ordinary ones. LOSH measures the spread of values "
             "around the local mean, scaled so that 1 is the study-area average: above 1 is a heterogeneous "
             "neighbourhood, below 1 a uniform one."),
            ("The typology.", "Ord &amp; Getis suggest reading H<sub>i</sub> with Gi*: a hot spot with low H is a "
             "coherent high-value area; a hot spot with high H is a high average produced by mixed values; high H at an "
             "average level marks a boundary between regimes."),
            ("Which residual.", "The paper prints the local residual about the focal unit's mean, \\(e_j = x_j - "
             "\\bar{x}_i\\). Both public implementations - spdep's <code>LOSH</code> and esda's <code>LOSH</code> - take "
             "the residual about each neighbour's own local mean, \\(e_j = x_j - \\bar{x}_j\\), for which the moments and "
             "the chi-square approximation are derived. This tool follows the implementations so its values can be "
             "checked against them."),
        ],
        "formulas": [
            ("Local mean and residual:", r"\bar{x}_j = \frac{\sum_k w_{jk} x_k}{\sum_k w_{jk}}, \qquad e_j = |x_j - \bar{x}_j|^{a}"),
            ("LOSH:", r"H_i = \frac{\sum_j w_{ij}\, e_j}{h_1 \sum_j w_{ij}}, \qquad h_1 = \frac{1}{n} \sum_j e_j"),
            ("Variance and chi-square approximation (a = 2):",
             r"\mathrm{Var}(H_i) = \frac{1}{n - 1} \Big(\frac{1}{h_1 W_{i1}}\Big)^2 (h_2 - h_1^2)\,(n W_{i2} - W_{i1}^2), \qquad "
             r"\frac{2 H_i}{\mathrm{Var}(H_i)} \sim \chi^2_{2/\mathrm{Var}(H_i)}",
             "\\(W_{i1} = \\sum_j w_{ij}\\), \\(W_{i2} = \\sum_j w_{ij}^2\\), \\(h_2 = \\frac{1}{n}\\sum_j e_j^2\\). For any "
             "exponent other than 2 the approximation is invalid and p-values are left NULL."),
        ],
        "note": "Binary weights; the window includes the unit itself by default (the Gi*-style window). Without it, H, its "
                "variance and the chi-square p-values match esda's <code>LOSH</code> exactly. The chi-square tail is "
                "computed with a stdlib incomplete-gamma routine, so scipy is not needed.",
        "params": [INPUT, ("FIELD", "Numeric Field", "—", "Must vary", "Variable")] + WEIGHTS + [
            ("EXPONENT", "Double", "2.0", "0.1–4", "Residual exponent a (2 = variance, 1 = absolute deviation)"),
            ("INCLUDE_SELF", "Boolean", "True", "—", "Include each unit in its own window"), MT, OUT_LAYER, REPORT],
        "outputs": [
            ("losh_h", "Double", "H<sub>i</sub> (1 = average local variability)"),
            ("losh_p", "Double", "Chi-square p-value (NULL unless a = 2)"),
            ("losh_padj", "Double", "Corrected p-value"),
            ("local_mean", "Double", "Local window mean"),
            ("gi_z", "Double", "Gi* z-score (binary weights, self included)"),
            ("losh_type", "String", "Mean/variance typology"),
        ],
        "report": "KPI row (median H, heterogeneous units, hot/cold spots); typology donut; scatter of Gi* z against H with "
                  "the H = 1 line.",
        "interp_title": "Reading the LOSH typology.",
        "interp": [
            ("Hot spot, homogeneous", "A coherent high-value area", "Treat as one area."),
            ("Hot spot, heterogeneous", "High average from mixed values", "Investigate at a finer grain before area-wide action."),
            ("Heterogeneous, average level", "A transition zone between regimes", "Look for a boundary on the map."),
        ],
        "see_also": ["alg-getis-ord", "alg-local-geary-c", "alg-gw-summary-stats"],
        "refs": [
            '<strong>Ord, J.K. &amp; Getis, A.</strong> (2012). "Local spatial heteroscedasticity (LOSH)." <em>The Annals '
            'of Regional Science</em> 48(2): 529–539. DOI: <a href="https://doi.org/10.1007/s00168-011-0492-y">'
            '10.1007/s00168-011-0492-y</a> — Defines H<sub>i</sub> and its chi-square approximation.',
            '<strong>Ord, J.K. &amp; Getis, A.</strong> (1995). "Local Spatial Autocorrelation Statistics: Distributional '
            'Issues and an Application." <em>Geographical Analysis</em> 27(4): 286–306. DOI: '
            '<a href="https://doi.org/10.1111/j.1538-4632.1995.tb00912.x">10.1111/j.1538-4632.1995.tb00912.x</a> — Gi*.',
            REF_BH],
    },
    {
        "id": "alg-partial-moran", "alg": "local_partial_moran", "after": "alg-losh",
        "title": "Partial and Conditional Local Moran (Clusters Net of Covariates)", "sidebar": "Partial / Conditional LISA", "icon": "🧮",
        "desc": "Does a cluster survive once you account for what already explains the field? Wolf's (2024) partial and "
                "conditional local Moran statistics answer it, and the report cross-tabulates the result against the "
                "plain LISA so the clusters the controls explain away are named.",
        "background": [
            ("Confounded local inference.", "A cluster of high rents may be nothing more than a cluster of central "
             "locations. Wolf (2024) shows that bivariate local statistics do not correct for this, and proposes two "
             "statistics built from a 'Moran-form' regression of the spatial lag on the variable and its confounders."),
            ("Conditional (full).", "The variable is regressed on the controls and Local Moran's I is computed on the "
             "residual. A remaining cluster is high beyond what the controls predict, next to neighbours that are also "
             "high beyond prediction. The variable of interest becomes the residual."),
            ("Partial.", "The lag Wy is regressed on [1, y, X]; the coefficient on y is the partial global Moran slope, and "
             "each unit's term in it - the corresponding row of \\((D'D)^{-1}D'\\) times its lag - is the local partial "
             "statistic. The variable of interest stays y; only the direct association with X is removed."),
        ],
        "formulas": [
            ("Conditional:", r"y = X\beta + e, \qquad I_i^{c} = (n - 1)\, \frac{e_i \sum_j w_{ij} e_j}{e'e}"),
            ("Partial:", r"D = [\mathbf{1},\, y,\, X], \qquad I_i^{p} = (n - 1)\, \big[D (D'D)^{-1}\big]_{i,y}\, (Wy)_i"),
            ("Labels:", r"\text{sign of the left component (centred) against the sign of } (Wy)_i"),
        ],
        "note": "Variables are centred and unit-scaled. Both statistics and the partial labels equal esda's "
                "<code>MoranLocalPartial</code> and <code>MoranLocalConditional</code> to floating-point precision. The "
                "tool refuses collinear controls (the regression would be undefined).",
        "params": [INPUT,
                   ("FIELD", "Numeric Field", "—", "Must vary", "Variable to analyse"),
                   ("CONTROLS", "Numeric Fields", "—", "≥ 1, not collinear", "Covariates to account for"),
                   ("MODE", "Enum", "Conditional", "Conditional / Partial", "Statistic"),
                   ] + WEIGHTS + [PERM, SEED, MT, OUT_LAYER, REPORT],
        "outputs": [
            ("pm_assoc", "Double", "Conditional or partial local statistic"),
            ("pm_p", "Double", "Two-sided conditional-permutation p-value"),
            ("pm_padj", "Double", "Corrected p-value"),
            ("quadrant", "String", "Class after accounting for the controls"),
            ("plain_quad", "String", "Class from the plain Local Moran's I"),
            ("pm_nbrs", "Integer", "Neighbours used"),
        ],
        "report": "KPI row (significant after controls, significant in the plain LISA, explained away, controls R²); class "
                  "donut; plain-versus-adjusted cross-table.",
        "interp_title": "Reading partial and conditional LISAs.",
        "interp": [
            ("Cluster survives", "Spatial structure the controls do not account for", "Look for an omitted factor or spillover."),
            ("Cluster explained away", "The controls account for it in the regression sense", "Not proof of causation."),
            ("Nothing significant after controls with high R²", "The controls absorb the pattern", "Model the field with OLS / spatial regression."),
        ],
        "see_also": ["alg-local-moran", "alg-ols", "alg-lm-diagnostics"],
        "refs": [
            '<strong>Wolf, L.J.</strong> (2024). "Confounded Local Inference: Extending Local Moran Statistics to Handle '
            'Confounding." <em>Annals of the American Association of Geographers</em> 114(6): 1216–1231. DOI: '
            '<a href="https://doi.org/10.1080/24694452.2024.2326541">10.1080/24694452.2024.2326541</a> — Defines both statistics.',
            REF_ANSELIN_1995, REF_BH],
    },
]
