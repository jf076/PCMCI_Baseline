import importlib
import time

import numpy as np

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ============================================================
# 1. Import SCM generator
# ============================================================

scm_generator = importlib.import_module(
    "01_generate_linear_scm"
)

generate_linear_scm = (
    scm_generator.generate_linear_scm
)


# ============================================================
# 2. Basic utility
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Safe division for evaluation metrics.
    """

    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
    )


# ============================================================
# 3. Read contemporaneous pattern from Tigramite graph
# ============================================================

def get_contemporaneous_pattern(
    graph,
    i,
    j,
):
    """
    Return the contemporaneous edge pattern
    from the perspective i --- j with i < j.

    Typical PCMCI+ patterns:

        -->
        <--
        o-o
        x-x

    Tigramite normally stores the reverse entry
    symmetrically. This helper also handles the
    case where only graph[j, i, 0] is populated.
    """

    link_ij = graph[
        i,
        j,
        0,
    ]

    if link_ij != "":
        return link_ij

    link_ji = graph[
        j,
        i,
        0,
    ]

    if link_ji == "":
        return ""

    # Reverse the representation so that
    # it is expressed from i to j.
    reverse_map = {
        "-->": "<--",
        "<--": "-->",
        "o-o": "o-o",
        "x-x": "x-x",
        "o->": "<-o",
        "<-o": "o->",
    }

    return reverse_map.get(
        link_ji,
        link_ji,
    )


# ============================================================
# 4. Extract Ground Truth sets
# ============================================================

def extract_true_sets(
    true_edges,
):
    """
    Convert true_edges into sets suitable for evaluation.

    Returns
    -------
    true_auto
        (source, target, lag)

    true_lagged_cross
        (source, target, lag)

    true_contemp_adj
        unordered pair (min(i,j), max(i,j))

    true_contemp_dir
        directed pair (source, target)
    """

    true_auto = set()

    true_lagged_cross = set()

    true_contemp_adj = set()

    true_contemp_dir = set()

    for edge in true_edges:

        source = int(
            edge["source"]
        )

        target = int(
            edge["target"]
        )

        lag = int(
            edge["lag"]
        )

        edge_type = (
            edge["edge_type"]
        )

        # ----------------------------------------------------
        # Autodependency
        # ----------------------------------------------------

        if edge_type == "auto":

            true_auto.add(
                (
                    source,
                    target,
                    lag,
                )
            )

        # ----------------------------------------------------
        # Lagged cross-link
        # ----------------------------------------------------

        elif (
            edge_type
            == "lagged_cross"
        ):

            true_lagged_cross.add(
                (
                    source,
                    target,
                    lag,
                )
            )

        # ----------------------------------------------------
        # Contemporaneous
        # ----------------------------------------------------

        elif (
            edge_type
            == "contemporaneous"
        ):

            pair = (
                min(
                    source,
                    target,
                ),
                max(
                    source,
                    target,
                ),
            )

            true_contemp_adj.add(
                pair
            )

            true_contemp_dir.add(
                (
                    source,
                    target,
                )
            )

        else:

            raise ValueError(
                "Unknown Ground Truth "
                f"edge_type: {edge_type}"
            )

    return {
        "auto":
            true_auto,

        "lagged_cross":
            true_lagged_cross,

        "contemp_adj":
            true_contemp_adj,

        "contemp_dir":
            true_contemp_dir,
    }


# ============================================================
# 5. Extract estimated sets
# ============================================================

def extract_estimated_sets(
    graph,
):
    """
    Convert Tigramite graph matrix into evaluation sets.
    """

    n_variables = (
        graph.shape[0]
    )

    tau_max = (
        graph.shape[2] - 1
    )

    estimated_auto = set()

    estimated_lagged_cross = set()

    estimated_contemp_adj = set()

    estimated_contemp_dir = set()

    unresolved_contemp = set()

    conflicting_contemp = set()

    other_contemp = set()

    # ========================================================
    # A. tau > 0 links
    # ========================================================

    for tau in range(
        1,
        tau_max + 1,
    ):

        for source in range(
            n_variables
        ):

            for target in range(
                n_variables
            ):

                link = graph[
                    source,
                    target,
                    tau,
                ]

                if link == "":
                    continue

                edge = (
                    source,
                    target,
                    tau,
                )

                # Autodependency
                if source == target:

                    estimated_auto.add(
                        edge
                    )

                # Lagged cross-link
                else:

                    estimated_lagged_cross.add(
                        edge
                    )

    # ========================================================
    # B. tau = 0 links
    # ========================================================

    for i in range(
        n_variables
    ):

        for j in range(
            i + 1,
            n_variables,
        ):

            pattern = (
                get_contemporaneous_pattern(
                    graph=graph,
                    i=i,
                    j=j,
                )
            )

            if pattern == "":
                continue

            pair = (
                i,
                j,
            )

            estimated_contemp_adj.add(
                pair
            )

            # ------------------------------------------------
            # Fully directed
            # ------------------------------------------------

            if pattern == "-->":

                estimated_contemp_dir.add(
                    (
                        i,
                        j,
                    )
                )

            elif pattern == "<--":

                estimated_contemp_dir.add(
                    (
                        j,
                        i,
                    )
                )

            # ------------------------------------------------
            # Unresolved orientation
            # ------------------------------------------------

            elif pattern == "o-o":

                unresolved_contemp.add(
                    pair
                )

            # ------------------------------------------------
            # Conflicting orientation
            # ------------------------------------------------

            elif pattern == "x-x":

                conflicting_contemp.add(
                    pair
                )

            # ------------------------------------------------
            # Other partially oriented patterns
            # ------------------------------------------------

            else:

                other_contemp.add(
                    (
                        pair,
                        pattern,
                    )
                )

    return {
        "auto":
            estimated_auto,

        "lagged_cross":
            estimated_lagged_cross,

        "contemp_adj":
            estimated_contemp_adj,

        "contemp_dir":
            estimated_contemp_dir,

        "unresolved_contemp":
            unresolved_contemp,

        "conflicting_contemp":
            conflicting_contemp,

        "other_contemp":
            other_contemp,
    }


# ============================================================
# 6. Build candidate edge spaces
# ============================================================

def build_candidate_sets(
    n_variables,
    tau_max,
):
    """
    Build all candidate adjacencies against which
    false positive rates are calculated.
    """

    # ========================================================
    # A. Autodependency candidates
    #
    # Xi(t-tau) -> Xi(t)
    # ========================================================

    auto_candidates = {
        (
            i,
            i,
            tau,
        )

        for i in range(
            n_variables
        )

        for tau in range(
            1,
            tau_max + 1,
        )
    }

    # ========================================================
    # B. Lagged cross-link candidates
    #
    # Xi(t-tau) -> Xj(t)
    # i != j
    # ========================================================

    lagged_cross_candidates = {
        (
            source,
            target,
            tau,
        )

        for source in range(
            n_variables
        )

        for target in range(
            n_variables
        )

        if source != target

        for tau in range(
            1,
            tau_max + 1,
        )
    }

    # ========================================================
    # C. Contemporaneous adjacency candidates
    #
    # unordered Xi(t) -- Xj(t)
    # ========================================================

    contemp_candidates = {
        (
            i,
            j,
        )

        for i in range(
            n_variables
        )

        for j in range(
            i + 1,
            n_variables,
        )
    }

    return {
        "auto":
            auto_candidates,

        "lagged_cross":
            lagged_cross_candidates,

        "contemp_adj":
            contemp_candidates,
    }


# ============================================================
# 7. Binary adjacency metrics
# ============================================================

def calculate_adjacency_metrics(
    true_set,
    estimated_set,
    candidate_set,
):
    """
    Calculate TP, FP, FN, TN, TPR and FPR.
    """

    if not true_set.issubset(
        candidate_set
    ):

        raise ValueError(
            "Ground Truth contains edges "
            "outside candidate space."
        )

    if not estimated_set.issubset(
        candidate_set
    ):

        raise ValueError(
            "Estimated graph contains edges "
            "outside candidate space."
        )

    tp_set = (
        true_set
        & estimated_set
    )

    fp_set = (
        estimated_set
        - true_set
    )

    fn_set = (
        true_set
        - estimated_set
    )

    negative_set = (
        candidate_set
        - true_set
    )

    tn_set = (
        negative_set
        - estimated_set
    )

    tp = len(
        tp_set
    )

    fp = len(
        fp_set
    )

    fn = len(
        fn_set
    )

    tn = len(
        tn_set
    )

    tpr = safe_divide(
        tp,
        tp + fn,
    )

    fpr = safe_divide(
        fp,
        fp + tn,
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,

        "TPR": tpr,
        "FPR": fpr,

        "tp_set": tp_set,
        "fp_set": fp_set,
        "fn_set": fn_set,
        "tn_set": tn_set,

        "n_true":
            len(true_set),

        "n_estimated":
            len(estimated_set),

        "n_candidates":
            len(candidate_set),
    }


# ============================================================
# 8. Complete Runge-style graph evaluation
# ============================================================

def evaluate_graph(
    true_edges,
    graph,
    n_variables,
    tau_max,
):
    """
    Evaluate an estimated time-series graph.

    Adjacency metrics follow the categories used in
    Runge (2020):

        1. lagged cross-links
        2. contemporaneous adjacencies
        3. autodependencies

    In addition, this function currently computes a
    directed-DAG operational version of contemporaneous
    orientation precision / recall.

    See comments below regarding CPDAG equivalence.
    """

    true_sets = (
        extract_true_sets(
            true_edges
        )
    )

    estimated_sets = (
        extract_estimated_sets(
            graph
        )
    )

    candidate_sets = (
        build_candidate_sets(
            n_variables=n_variables,
            tau_max=tau_max,
        )
    )

    # ========================================================
    # A. Adjacency metrics
    # ========================================================

    auto_metrics = (
        calculate_adjacency_metrics(
            true_set=
                true_sets["auto"],

            estimated_set=
                estimated_sets["auto"],

            candidate_set=
                candidate_sets["auto"],
        )
    )

    lagged_cross_metrics = (
        calculate_adjacency_metrics(
            true_set=
                true_sets[
                    "lagged_cross"
                ],

            estimated_set=
                estimated_sets[
                    "lagged_cross"
                ],

            candidate_set=
                candidate_sets[
                    "lagged_cross"
                ],
        )
    )

    contemp_metrics = (
        calculate_adjacency_metrics(
            true_set=
                true_sets[
                    "contemp_adj"
                ],

            estimated_set=
                estimated_sets[
                    "contemp_adj"
                ],

            candidate_set=
                candidate_sets[
                    "contemp_adj"
                ],
        )
    )

    # ========================================================
    # B. Contemporaneous orientation
    # ========================================================

    true_directions = (
        true_sets[
            "contemp_dir"
        ]
    )

    estimated_directions = (
        estimated_sets[
            "contemp_dir"
        ]
    )

    correct_directions = (
        true_directions
        & estimated_directions
    )

    # Example:
    #
    # true: X1 -> X3
    # estimated: X3 -> X1
    #
    reversed_directions = {
        estimated_edge

        for estimated_edge
        in estimated_directions

        if (
            estimated_edge[1],
            estimated_edge[0],
        )
        in true_directions
    }

    n_correct_direction = len(
        correct_directions
    )

    n_true_contemp = len(
        true_sets[
            "contemp_adj"
        ]
    )

    n_estimated_contemp = len(
        estimated_sets[
            "contemp_adj"
        ]
    )

    # --------------------------------------------------------
    # Operational orientation recall:
    #
    # correctly directed true contemporaneous links
    # -----------------------------------------------
    # total true contemporaneous links
    # --------------------------------------------------------

    orientation_recall = (
        safe_divide(
            n_correct_direction,
            n_true_contemp,
        )
    )

    # --------------------------------------------------------
    # Operational orientation precision:
    #
    # correctly directed estimated links
    # -----------------------------------
    # all estimated contemporaneous adjacencies
    #
    # Therefore:
    # - reverse direction
    # - unresolved o-o
    # - conflicting x-x
    # - false adjacency
    #
    # all reduce this precision.
    # --------------------------------------------------------

    orientation_precision = (
        safe_divide(
            n_correct_direction,
            n_estimated_contemp,
        )
    )

    # ========================================================
    # C. Conflict rate
    # ========================================================

    n_conflicts = len(
        estimated_sets[
            "conflicting_contemp"
        ]
    )

    conflict_rate = (
        safe_divide(
            n_conflicts,
            n_estimated_contemp,
        )
    )

    return {
        "auto":
            auto_metrics,

        "lagged_cross":
            lagged_cross_metrics,

        "contemporaneous":
            contemp_metrics,

        "orientation": {

            "correct":
                n_correct_direction,

            "correct_set":
                correct_directions,

            "reversed":
                len(
                    reversed_directions
                ),

            "reversed_set":
                reversed_directions,

            "unresolved":
                len(
                    estimated_sets[
                        "unresolved_contemp"
                    ]
                ),

            "unresolved_set":
                estimated_sets[
                    "unresolved_contemp"
                ],

            "conflicts":
                n_conflicts,

            "conflict_set":
                estimated_sets[
                    "conflicting_contemp"
                ],

            "other_patterns":
                estimated_sets[
                    "other_contemp"
                ],

            "recall":
                orientation_recall,

            "precision":
                orientation_precision,

            "conflict_rate":
                conflict_rate,
        },
    }


# ============================================================
# 9. Formatting helpers
# ============================================================

def format_lagged_edge(
    edge,
    var_names,
):
    """
    Format (source, target, lag).
    """

    source, target, lag = (
        edge
    )

    return (
        f"{var_names[source]}"
        f"(t-{lag}) "
        f"-> "
        f"{var_names[target]}(t)"
    )


def format_contemp_pair(
    pair,
    var_names,
):
    """
    Format unordered contemporaneous adjacency.
    """

    i, j = pair

    return (
        f"{var_names[i]}(t) "
        f"-- "
        f"{var_names[j]}(t)"
    )


def format_contemp_direction(
    edge,
    var_names,
):
    """
    Format directed contemporaneous edge.
    """

    source, target = edge

    return (
        f"{var_names[source]}(t) "
        f"-> "
        f"{var_names[target]}(t)"
    )


# ============================================================
# 10. Print report
# ============================================================

def print_metric_report(
    report,
    var_names,
):
    """
    Print evaluation report.
    """

    print(
        "\nRunge-style Evaluation"
    )

    print("=" * 78)

    # ========================================================
    # A. Adjacency table
    # ========================================================

    print(
        "\n[Adjacency metrics]"
    )

    header = (
        f"{'Category':<24}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'FN':>6}"
        f"{'TN':>6}"
        f"{'TPR':>12}"
        f"{'FPR':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    rows = [
        (
            "Lagged cross",
            report[
                "lagged_cross"
            ],
        ),
        (
            "Autodependency",
            report[
                "auto"
            ],
        ),
        (
            "Contemporaneous",
            report[
                "contemporaneous"
            ],
        ),
    ]

    for name, metrics in rows:

        print(
            f"{name:<24}"
            f"{metrics['TP']:>6}"
            f"{metrics['FP']:>6}"
            f"{metrics['FN']:>6}"
            f"{metrics['TN']:>6}"
            f"{metrics['TPR']:>12.4f}"
            f"{metrics['FPR']:>12.4f}"
        )

    # ========================================================
    # B. Orientation metrics
    # ========================================================

    orientation = (
        report[
            "orientation"
        ]
    )

    print(
        "\n[Contemporaneous orientation]"
    )

    print(
        "correctly directed =",
        orientation[
            "correct"
        ],
    )

    print(
        "reversed direction =",
        orientation[
            "reversed"
        ],
    )

    print(
        "unresolved (o-o)   =",
        orientation[
            "unresolved"
        ],
    )

    print(
        "conflicts (x-x)    =",
        orientation[
            "conflicts"
        ],
    )

    print(
        "orientation recall = "
        f"{orientation['recall']:.4f}"
    )

    print(
        "orientation precision = "
        f"{orientation['precision']:.4f}"
    )

    print(
        "conflict rate = "
        f"{orientation['conflict_rate']:.4f}"
    )

    # ========================================================
    # C. Diagnostic error lists
    # ========================================================

    print(
        "\n[Diagnostic errors]"
    )

    # --------------------------------------------------------
    # Missed lagged cross-links
    # --------------------------------------------------------

    missed_lagged = (
        report[
            "lagged_cross"
        ][
            "fn_set"
        ]
    )

    print(
        "\nMissed lagged cross-links:"
    )

    if missed_lagged:

        for edge in sorted(
            missed_lagged
        ):

            print(
                "  ",
                format_lagged_edge(
                    edge,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # False lagged cross-links
    # --------------------------------------------------------

    false_lagged = (
        report[
            "lagged_cross"
        ][
            "fp_set"
        ]
    )

    print(
        "\nFalse lagged cross-links:"
    )

    if false_lagged:

        for edge in sorted(
            false_lagged
        ):

            print(
                "  ",
                format_lagged_edge(
                    edge,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # False autodependencies
    # --------------------------------------------------------

    false_auto = (
        report[
            "auto"
        ][
            "fp_set"
        ]
    )

    print(
        "\nFalse autodependencies:"
    )

    if false_auto:

        for edge in sorted(
            false_auto
        ):

            print(
                "  ",
                format_lagged_edge(
                    edge,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Missed contemporaneous adjacencies
    # --------------------------------------------------------

    missed_contemp = (
        report[
            "contemporaneous"
        ][
            "fn_set"
        ]
    )

    print(
        "\nMissed contemporaneous adjacencies:"
    )

    if missed_contemp:

        for pair in sorted(
            missed_contemp
        ):

            print(
                "  ",
                format_contemp_pair(
                    pair,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # False contemporaneous adjacencies
    # --------------------------------------------------------

    false_contemp = (
        report[
            "contemporaneous"
        ][
            "fp_set"
        ]
    )

    print(
        "\nFalse contemporaneous adjacencies:"
    )

    if false_contemp:

        for pair in sorted(
            false_contemp
        ):

            print(
                "  ",
                format_contemp_pair(
                    pair,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Reversed true directions
    # --------------------------------------------------------

    reversed_directions = (
        orientation[
            "reversed_set"
        ]
    )

    print(
        "\nReversed contemporaneous directions:"
    )

    if reversed_directions:

        for edge in sorted(
            reversed_directions
        ):

            print(
                "  ",
                format_contemp_direction(
                    edge,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Unresolved
    # --------------------------------------------------------

    unresolved = (
        orientation[
            "unresolved_set"
        ]
    )

    print(
        "\nUnresolved contemporaneous links:"
    )

    if unresolved:

        for pair in sorted(
            unresolved
        ):

            print(
                "  ",
                format_contemp_pair(
                    pair,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )

    # --------------------------------------------------------
    # Conflicts
    # --------------------------------------------------------

    conflicts = (
        orientation[
            "conflict_set"
        ]
    )

    print(
        "\nConflicting contemporaneous links:"
    )

    if conflicts:

        for pair in sorted(
            conflicts
        ):

            print(
                "  ",
                format_contemp_pair(
                    pair,
                    var_names,
                ),
            )

    else:

        print(
            "  None"
        )


# ============================================================
# 11. Main
# ============================================================

def main():

    # ========================================================
    # Runge (2020) default linear Gaussian setup
    # ========================================================

    N = 5
    T = 500

    A_MAX = 0.95

    TAU_MAX = 5

    PC_ALPHA = 0.01

    SEED = 0

    BURN_IN = 500

    VAR_NAMES = [
        f"X{i}"
        for i in range(N)
    ]

    print(
        "Runge (2020) - "
        "PCMCI+ metric evaluation"
    )

    print("=" * 70)

    # ========================================================
    # 12. Generate exactly the same seed=0 dataset
    # ========================================================

    (
        data,
        true_edges,
        model_info,
    ) = generate_linear_scm(
        n_variables=N,
        n_samples=T,
        a_max=A_MAX,
        tau_max=TAU_MAX,
        seed=SEED,
        burn_in=BURN_IN,
    )

    print(
        "\nDataset:"
    )

    print(
        "shape =",
        data.shape,
    )

    print(
        "spectral radius = "
        f"{model_info['spectral_radius']:.6f}"
    )

    # ========================================================
    # 13. Run PCMCI+
    # ========================================================

    dataframe = pp.DataFrame(
        data=data,
        var_names=VAR_NAMES,
    )

    parcorr = ParCorr()

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=parcorr,
        verbosity=0,
    )

    start_time = (
        time.perf_counter()
    )

    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA,
        contemp_collider_rule="majority",
        conflict_resolution=True,
        reset_lagged_links=False,
        max_combinations=1,
        fdr_method="none",
    )

    runtime = (
        time.perf_counter()
        - start_time
    )

    graph = (
        results["graph"]
    )

    # ========================================================
    # 14. Evaluate
    # ========================================================

    report = evaluate_graph(
        true_edges=true_edges,
        graph=graph,
        n_variables=N,
        tau_max=TAU_MAX,
    )

    print_metric_report(
        report=report,
        var_names=VAR_NAMES,
    )

    print(
        "\n[Runtime]"
    )

    print(
        f"PCMCI+ runtime = "
        f"{runtime:.4f} s"
    )

    print(
        "\n03 metric evaluation completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()