# -*- coding: utf-8 -*-
"""Manual cards for v3.12.0 - regionalisation and partition quality."""

INPUT = ("INPUT", "Vector Layer", "—", "Polygons for contiguity; any geometry for KNN / distance", "Units to aggregate")
WEIGHTS = [
    ("WEIGHT_TYPE", "Enum", "Queen contiguity", "Queen / Rook / KNN / Distance Band", "Which units count as touching"),
    ("KNN", "Integer", "8", "≥ 1", "Neighbours (KNN only)"),
    ("DISTANCE_BAND", "Double", "1000.0", "> 0", "Distance band (Distance Band only)"),
]
FIELDS = ("FIELDS", "Numeric Fields", "—", "≥ 1; standardised internally", "Attributes the regions should be homogeneous on")
OUT = ("OUTPUT", "Vector Layer", "—", "Optional path; styled by region", "Input features with region_id and silhouettes")
REPORT = ("HTML_REPORT", "File", "temporary", "*.html", "Analyst report with the partition-quality assessment")
REGION_OUTPUTS = [
    ("region_id", "Integer", "Region (0-based); NULL where a unit could not be assigned"),
    ("sil", "Double", "Attribute silhouette of the unit"),
    ("bnd_sil", "Double", "Boundary silhouette (0 for interior units)"),
    ("path_sil", "Double", "Silhouette on shortest-path distance through the neighbourhood graph (NULL above 3000 units)"),
]
QUALITY_REPORT = ("The shared partition-quality block: regions, R², mean attribute, boundary and path silhouettes, size "
                  "balance; a per-region table (units, SSD, mean silhouette, isoperimetric compactness of the dissolved "
                  "region, contiguity); the regions with the weakest silhouette as a bar chart. The layer is styled with "
                  "one colour per region.")
REF_DAR = ('<strong>Duque, J.C., Anselin, L. &amp; Rey, S.J.</strong> (2012). "The max-p-regions problem." <em>Journal of '
           'Regional Science</em> 52(3): 397–419. DOI: <a href="https://doi.org/10.1111/j.1467-9787.2011.00743.x">'
           '10.1111/j.1467-9787.2011.00743.x</a> — Defines the problem and the first heuristic.')
REF_WRK = ('<strong>Wei, R., Rey, S. &amp; Knaap, E.</strong> (2021). "Efficient regionalization for spatially explicit '
           'neighborhood delineation." <em>International Journal of Geographical Information Science</em> 35(1): 135–151. '
           'DOI: <a href="https://doi.org/10.1080/13658816.2020.1759806">10.1080/13658816.2020.1759806</a> — The '
           'construction-plus-simulated-annealing heuristic used here.')
REF_OPEN = ('<strong>Openshaw, S.</strong> (1977). "A geographical solution to scale and aggregation problems in '
            'region-building, partitioning and spatial modelling." <em>Transactions of the Institute of British '
            'Geographers</em> 2(4): 459–472. DOI: <a href="https://doi.org/10.2307/622300">10.2307/622300</a> — AZP.')
REF_OR = ('<strong>Openshaw, S. &amp; Rao, L.</strong> (1995). "Algorithms for reengineering 1991 census geography." '
          '<em>Environment and Planning A</em> 27(3): 425–446. DOI: <a href="https://doi.org/10.1068/a270425">'
          '10.1068/a270425</a> — AZP with simulated annealing and tabu search.')
REF_GUO = ('<strong>Guo, D.</strong> (2008). "Regionalization with dynamically constrained agglomerative clustering and '
           'partitioning (REDCAP)." <em>International Journal of Geographical Information Science</em> 22(7): 801–823. '
           'DOI: <a href="https://doi.org/10.1080/13658810701674970">10.1080/13658810701674970</a> — REDCAP.')
REF_AGNM = ('<strong>Assunção, R.M., Neves, M.C., Câmara, G. &amp; da Costa Freitas, C.</strong> (2006). "Efficient '
            'regionalization techniques for socio-economic geographical units using minimum spanning trees." '
            '<em>International Journal of Geographical Information Science</em> 20(7): 797–811. DOI: '
            '<a href="https://doi.org/10.1080/13658810600665111">10.1080/13658810600665111</a> — SKATER, the tree-cut rule '
            'REDCAP stage 2 shares.')
REF_ROUS = ('<strong>Rousseeuw, P.J.</strong> (1987). "Silhouettes: A graphical aid to the interpretation and validation '
            'of cluster analysis." <em>Journal of Computational and Applied Mathematics</em> 20: 53–65. DOI: '
            '<a href="https://doi.org/10.1016/0377-0427(87)90125-7">10.1016/0377-0427(87)90125-7</a> — The silhouette.')
REF_WKR = ('<strong>Wolf, L.J., Knaap, E. &amp; Rey, S.</strong> (2021). "Geosilhouettes: Geographical measures of cluster '
           'fit." <em>Environment and Planning B: Urban Analytics and City Science</em> 48(3): 521–539. DOI: '
           '<a href="https://doi.org/10.1177/2399808319875752">10.1177/2399808319875752</a> — Boundary and path silhouettes.')
REF_RH = ('<strong>Rosenberg, A. &amp; Hirschberg, J.</strong> (2007). "V-Measure: A conditional entropy-based external '
          'cluster evaluation measure." <em>Proceedings of EMNLP-CoNLL 2007</em>, pp. 410–420. — Homogeneity, '
          'completeness and the V-measure.')

QUALITY_FORMULAS = [
    ("Within-region heterogeneity (objective):", r"SSD = \sum_{r} \sum_{i \in r} \lVert z_i - \bar{z}_r \rVert^2, \qquad R^2 = 1 - SSD / \sum_i \lVert z_i \rVert^2"),
    ("Silhouette of unit i:", r"s_i = \frac{b_i - a_i}{\max(a_i, b_i)}",
     "\\(a_i\\): mean distance to the other members of its region; \\(b_i\\): smallest mean distance to another region."),
]

CARDS = [
    {
        "id": "alg-max-p", "alg": "max_p_regions", "after": "alg-skater",
        "title": "Max-p Regions (Zones Meeting a Minimum Size)", "sidebar": "Max-p Regions", "icon": "🗺️",
        "desc": "As many contiguous regions as possible such that every region reaches a floor - a minimum population, "
                "area or count - and, among those, the most homogeneous on the attribute fields. The number of regions "
                "is an output, not an input: the planning workhorse for 'the smallest districts that each still serve "
                "at least N people'.",
        "background": [
            ("Why a floor instead of k.", "SKATER, K-means, AZP and REDCAP all need the number of regions in advance. "
             "Many planning questions fix something else: every school catchment needs at least 1500 pupils, every "
             "reporting unit at least 5000 residents for a stable rate. Duque, Anselin &amp; Rey (2012) formalised this "
             "as the max-p-regions problem: maximise the number of contiguous regions subject to the floor, then "
             "minimise heterogeneity among the solutions with that many regions."),
            ("Heuristic.", "The problem is NP-hard. Following Wei, Rey &amp; Knaap (2021), a construction phase grows "
             "regions from random seeds - each growing region takes the adjacent unassigned unit closest to its attribute "
             "centroid until the floor is met - and repeats this many times, keeping the run with the most regions. Units "
             "that cannot join a region meeting the floor are attached to the neighbouring region whose heterogeneity "
             "rises least. A local search then moves units across borders (simulated annealing by default) while every "
             "region stays contiguous and above the floor."),
        ],
        "formulas": [
            ("Problem:", r"\max p \quad \text{s.t.} \quad \sum_{i \in R_r} f_i \ge T \;\; \forall r, \quad R_r \text{ contiguous}, \quad \bigcup_r R_r = \text{assignable units}"),
            ("Tie-break among maximum-p partitions:", r"\min \; SSD"),
        ] + QUALITY_FORMULAS,
        "note": "<code>core/regionalization_engines.py::max_p_regions</code>, numpy only - the plan had this arrive through the "
                "optional <code>spopt</code> package; writing it here keeps the tool working on a stock QGIS. Units in a "
                "connected part of the map whose total floor is below the threshold cannot form any region and are left "
                "unassigned. The report states the upper bound floor(total / T) next to the p found.",
        "params": [INPUT, FIELDS,
                   ("FLOOR_FIELD", "Numeric Field", "—", "≥ 0", "Spatially extensive variable summed per region"),
                   ("THRESHOLD", "Double", "10000", "> 0", "Minimum floor per region"),
                   ] + WEIGHTS + [
                   ("ITERATIONS", "Integer", "99", "1–5000", "Construction runs"),
                   ("LOCAL_SEARCH", "Enum", "Simulated annealing", "SA / Tabu / Greedy / None", "Improvement phase"),
                   ("RANDOM_SEED", "Integer", "42", "≥ 0", "Seed"), OUT, REPORT],
        "outputs": REGION_OUTPUTS,
        "report": "KPI row (p, smallest region floor, SSD before and after the local search, unassigned units), the upper "
                  "bound on p, the per-region floor sums, and " + QUALITY_REPORT[0].lower() + QUALITY_REPORT[1:],
        "interp_title": "Reading a max-p solution.",
        "interp": [
            ("p well below floor(total/T)", "Contiguity or the attribute objective costs regions", "Try more construction runs; inspect large regions."),
            ("Unassigned units", "A part of the map cannot reach the floor alone", "Lower the floor or merge those units by hand."),
            ("Same p across seeds", "The maximum is probably found", "Compare SSD across seeds and keep the lowest."),
        ],
        "see_also": ["alg-skater", "alg-azp", "alg-region-quality"],
        "refs": [REF_DAR, REF_WRK, REF_ROUS],
    },
    {
        "id": "alg-azp", "alg": "azp_regionalization", "after": "alg-max-p",
        "title": "AZP Regionalization (Automatic Zoning Procedure)", "sidebar": "AZP Regionalization", "icon": "🧭",
        "desc": "Exactly k contiguous regions, as homogeneous as possible, by Openshaw's Automatic Zoning Procedure: start "
                "from a random contiguous partition and keep moving units across region borders while the move lowers "
                "the within-region sum of squares and leaves the donor region connected. Basic, simulated-annealing and "
                "tabu variants; several random starts, best kept.",
        "background": [
            ("The zone-design classic.", "Openshaw (1977) introduced AZP to show that the modifiable areal unit problem "
             "can be turned around: if results depend on the zoning, design the zoning for the purpose. AZP is a local "
             "search over contiguous partitions with a fixed number of regions."),
            ("Escaping local optima.", "Basic AZP accepts only improving moves and stops at the first local optimum. "
             "Openshaw &amp; Rao (1995) added simulated annealing (accept a worsening move with probability "
             "\\(e^{-\\Delta/T}\\), with T cooling) and tabu search (always make the best admissible move and forbid "
             "reversing recent moves for a while). Several random starts further reduce the dependence on the start."),
        ],
        "formulas": [
            ("Objective:", r"\min_{R_1,\dots,R_k \text{ contiguous}} \; SSD"),
            ("Move of unit i from region A to a neighbouring region B is admissible if:", r"A \setminus \{i\} \text{ is non-empty and connected}"),
            ("Simulated-annealing acceptance:", r"P(\text{accept}) = \min\big(1, e^{-\Delta SSD / T}\big), \qquad T \leftarrow 0.95\,T"),
        ] + QUALITY_FORMULAS[1:],
        "note": "<code>core/regionalization_engines.py::azp</code>. The change in SSD of a move is computed from running "
                "region sums in O(p), so a full sweep costs O(n p) plus the connectivity check of the donor region. A "
                "neighbourhood graph with more disconnected parts than k is refused.",
        "params": [INPUT, FIELDS,
                   ("K_REGIONS", "Integer", "5", "1 ≤ k ≤ n", "Number of regions"),
                   ("METHOD", "Enum", "Simulated annealing", "Basic / SA / Tabu", "Local-search variant"),
                   ] + WEIGHTS + [
                   ("RESTARTS", "Integer", "10", "1–500", "Random starts"),
                   ("MIN_SIZE", "Integer", "1", "≥ 1", "Minimum units per region"),
                   ("RANDOM_SEED", "Integer", "42", "≥ 0", "Seed"), OUT, REPORT],
        "outputs": REGION_OUTPUTS,
        "report": "KPI row (regions, final SSD, median SSD of the random starts and the improvement) and " +
                  QUALITY_REPORT[0].lower() + QUALITY_REPORT[1:],
        "interp_title": "Reading an AZP solution.",
        "interp": [
            ("Large improvement over the random starts", "The search did real work", "Check stability with another seed."),
            ("Boundaries change between seeds", "Several near-equivalent zonings exist", "Adopt boundaries common to most runs."),
            ("Single-unit regions", "Outliers captured as their own region", "Raise the minimum size."),
        ],
        "see_also": ["alg-max-p", "alg-redcap", "alg-region-quality"],
        "refs": [REF_OPEN, REF_OR, REF_ROUS],
    },
    {
        "id": "alg-redcap", "alg": "redcap_regionalization", "after": "alg-azp",
        "title": "REDCAP Regionalization (Contiguity-Constrained Clustering)", "sidebar": "REDCAP Regionalization", "icon": "🌳",
        "desc": "Guo's (2008) REDCAP: a contiguity-constrained hierarchical clustering builds a spanning tree by merging "
                "the most similar touching clusters first, and the tree is then cut into k regions where cutting most "
                "reduces heterogeneity. Deterministic - the same input always gives the same regions.",
        "background": [
            ("Two stages.", "Stage 1 starts from single units and repeatedly merges the two <em>adjacent</em> clusters "
             "with the smallest linkage distance, adding to a spanning tree the shortest contiguity edge between them. "
             "Stage 2 removes k-1 tree edges, each time the one whose removal most reduces the within-region sum of "
             "squares - the same rule as SKATER."),
            ("Full-order linkage.", "SKATER's tree is a minimum spanning tree of the contiguity graph, built from "
             "neighbouring pairs only. REDCAP's linkage is computed over all unit pairs of the two clusters "
             "(single, average, complete or Ward), updated by the Lance–Williams formulas. Guo's experiments found "
             "full-order complete linkage the most homogeneous; it is the default."),
        ],
        "formulas": [
            ("Merge rule (stage 1):", r"(A, B) = \arg\min_{A \sim B} \; d(A, B), \qquad A \sim B \iff \exists\, i \in A, j \in B: w_{ij} = 1"),
            ("Lance–Williams updates:", r"d_{\text{complete}}(A \cup B, C) = \max(d_{AC}, d_{BC}), \qquad d_{\text{average}} = \frac{n_A d_{AC} + n_B d_{BC}}{n_A + n_B}"),
            ("Cut rule (stage 2):", r"e^* = \arg\max_{e \in T} \big[SSD(R) - SSD(R_1^e) - SSD(R_2^e)\big]"),
        ] + QUALITY_FORMULAS[1:],
        "note": "<code>core/regionalization_engines.py::redcap</code>. Keeps an n × n distance matrix, so the tool refuses "
                "more than 4000 units (use Max-p or AZP there). The graph must be connected.",
        "params": [INPUT, FIELDS,
                   ("K_REGIONS", "Integer", "5", "1 ≤ k ≤ n", "Number of regions"),
                   ("LINKAGE", "Enum", "Full-order complete", "Complete / Average / Ward / Single", "Cluster linkage"),
                   ] + WEIGHTS + [("MIN_SIZE", "Integer", "1", "≥ 1", "Minimum units per region"), OUT, REPORT],
        "outputs": REGION_OUTPUTS,
        "report": "KPI row (regions, SSD, linkage) and " + QUALITY_REPORT[0].lower() + QUALITY_REPORT[1:],
        "interp_title": "Reading a REDCAP solution.",
        "interp": [
            ("Long thin regions", "Chaining (typical of single linkage)", "Switch to complete or Ward linkage."),
            ("Very unequal region sizes", "The tree cut isolated a distinct pocket early", "Raise the minimum size."),
            ("Weak boundary silhouettes", "Boundaries fixed early are never revisited", "Refine with AZP."),
        ],
        "see_also": ["alg-skater", "alg-azp", "alg-region-quality"],
        "refs": [REF_GUO, REF_AGNM, REF_ROUS],
    },
    {
        "id": "alg-region-quality", "alg": "regionalization_quality", "after": "alg-redcap",
        "title": "Regionalization Quality Assessment", "sidebar": "Regionalization Quality", "icon": "📏",
        "desc": "Judges any partition of a layer - a region_id from SKATER, Max-p, AZP or REDCAP, a K-means or DBSCAN "
                "cluster, or an existing district field - against the attributes it is supposed to group: R², "
                "attribute, boundary and path silhouettes, size balance, contiguity and compactness, and agreement with "
                "a reference partition.",
        "background": [
            ("The ruler, not the cutter.", "A regionalisation that cannot be scored cannot be defended in a review or "
             "compared with another. These are the cluster-fit measures esda collects, applied the same way to every "
             "partition so two runs can be compared line by line."),
            ("Geographical silhouettes.", "The ordinary silhouette (Rousseeuw 1987) ignores space. Wolf, Knaap &amp; Rey "
             "(2021) add two geographical versions: the boundary silhouette asks the silhouette question only of units on "
             "a region edge and only against the regions they touch - the comparison that matters for a border - and "
             "the path silhouette measures distance along the neighbourhood graph, so a region scores well only if it is "
             "both alike and well connected."),
            ("Agreement with a reference.", "Homogeneity (every region inside one reference zone), completeness (every "
             "reference zone inside one region) and their harmonic mean, the V-measure (Rosenberg &amp; Hirschberg 2007), "
             "optionally weighted by area as in esda's areal versions."),
        ],
        "formulas": QUALITY_FORMULAS + [
            ("Boundary silhouette (boundary units only):", r"s_i^{B} = \frac{b_i^{B} - a_i}{\max(a_i, b_i^{B})}, \quad b_i^{B} = \min_{r \in \text{touching}(i)} \bar{d}(i, r)"),
            ("Homogeneity, completeness, V-measure:", r"h = 1 - \frac{H(C \mid K)}{H(C)}, \quad c = 1 - \frac{H(K \mid C)}{H(K)}, \quad V = \frac{2hc}{h + c}"),
            ("Compactness of a region:", r"IPQ = \frac{4\pi A}{P^2}"),
        ],
        "note": "<code>core/region_quality.py</code>, numpy only. Attribute, boundary and path silhouettes and the three "
                "entropy measures equal scikit-learn's and esda's on the same data to floating-point precision. The path "
                "silhouette needs all-pairs shortest paths and is skipped above 3000 units.",
        "params": [INPUT,
                   ("LABEL_FIELD", "Field", "—", "Text or number; ≥ 2 values", "Partition to assess"),
                   ("FIELDS", "Numeric Fields", "—", "≥ 1", "Attributes the partition should be homogeneous on"),
                   ("REFERENCE_FIELD", "Field", "—", "Optional", "Reference partition to compare with"),
                   ] + WEIGHTS + [("AREA_WEIGHTED", "Boolean", "False", "—", "Weight the reference comparison by area"),
                                  ("OUTPUT", "Vector Layer", "—", "Optional path", "Units with silhouette fields"), REPORT],
        "outputs": REGION_OUTPUTS[1:],
        "report": QUALITY_REPORT.replace("The layer is styled with one colour per region.",
                                         "With a reference field, homogeneity, completeness and V-measure. The layer is "
                                         "styled by attribute silhouette."),
        "interp_title": "Reading the quality figures.",
        "interp": [
            ("Higher R² and silhouette for partition A than B on the same fields", "A separates the attributes better",
             "Prefer A unless its regions are less compact or non-contiguous."),
            ("Negative boundary silhouettes along one border", "That border puts units in the wrong region", "Inspect it on the map."),
            ("High homogeneity, low completeness", "The regions subdivide the reference zones", "Expected when k is larger than the reference count."),
        ],
        "see_also": ["alg-skater", "alg-max-p", "alg-multivar-cluster"],
        "refs": [REF_ROUS, REF_WKR, REF_RH],
    },
]
