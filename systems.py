import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

kernels = {
            "standard":  np.array([[0.05, 0.2, 0.05],
                                   [ 0.2,  -1,  0.2],
                                   [0.05, 0.2, 0.05]], )
}

#http://mrob.com/pub/comp/xmorphia/pearson-classes.html
pearson = {
    'alpha_right': {
        'description': """
        Spatial-temporal chaos composed mainly of wavelets and "fledgling spirals" that repeatedly
        grow and/or multiply and quickly annihilate upon hitting another object.

        Type epsilon (ε) differs from this by supporting rings, spot mitosis, and longer-lived clumps of spots.""",
        'file_stub': "alpha(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.014,
            'k': 0.053
        },
        'scaling': {
            "a": {
                'icl': 0.1,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'α'
    },
    'alpha_left': {
        'description': """
        Spatial-temporal chaos composed mainly of wavelets and "fledgling spirals" that repeatedly
        grow and/or multiply and quickly annihilate upon hitting another object.

        Type epsilon (ε) differs from this by supporting rings, spot mitosis, and longer-lived clumps of spots.""",
        'file_stub': "alpha(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.010,
            'k': 0.047
        },
        'scaling': {
            "a": {
                'icl': 0.05,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'α'
    },
    'beta_right': {
        'description': """
        Spatial-temporal chaos with localised red and blue state in different spots at different
        times. This looks like waves on a blue ocean with periodic red "voids" that open up suddenly and then
        quickly fill in with blue.""",
        'file_stub': "beta(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.026,
            'k': 0.051
        },
        'scaling': {
            "a": {
                'icl': 0.35177,
                'im': 0.35180,
                'icu': 0.35184
            },
            "b": {
                'icl': 0.21886,
                'im': 0.218875,
                'icu': 0.21889
            },
        },
        'symbol': 'β'
    },
    'beta_left': {
        'description': """
        Spatial-temporal chaos with localised red and blue state in different spots at different
        times. This looks like waves on a blue ocean with periodic red "voids" that open up suddenly and then
        quickly fill in with blue. """,
        'file_stub': "beta(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.014,
            'k': 0.039
        },
        'scaling': {
            "a": {
                'icl': 0.27,
                'im': 0.28,
                'icu': 0.29
            },
            "b": {
                'icl': 0.18,
                'im': 0.19,
                'icu': .2
            },
        },
        'symbol': 'β'
    },
    'gamma_left': {
        'description': """
        Stripe either wormlike or branching, with endless instability in the form of
        occasional changes due to overcrowding, grain boundary instabilities,
        or other localised events, and oscillation in the ∂u/∂t component.""",
        'file_stub': "gamma(left)",
        'cmap': {'a':  "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.022,
            'k': 0.051
        },
        'scaling': {
            "a": {
                'icl': 0.27,
                'im': 0.42,
                'icu': 0.70
            },
            "b": {
                'icl': 0.17,
                'im': 0.23,
                'icu': 0.29
            },
        },
        'symbol': 'γ'
    },
    'gamma_right': {
        'description': """
        Stripe either wormlike or branching, with endless instability in the form of
        occasional changes due to overcrowding, grain boundary instabilities,
        or other localised events, and oscillation in the ∂u/∂t component.""",
        'file_stub': "gamma(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.026,
            'k': 0.055
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.55,
                'icu': 0.85
            },
            "b": {
                'icl': 0.0,
                'im': 0.17,
                'icu': 0.34
            },
        },
        'symbol': 'γ'
    },
    'delta_right': {
        'description': """
        Includes true Turing patterns and many parameter values that produce similar patterns
        through a presumably related effect. The true Turing pattern arises from a starting
        pattern that is all blue, with arbitrarily small noise or other irregularities.
        In "Turing-similar" systems, a pattern cannot arise from an all-blue starting state,
        instead they need to be "seeded" by a large amplitude disturbance (anything non-blue
        and at least as large as a typical negaton will usually suffice); and the pattern grows outward
        from the disturbance. In either case, the resulting pattern is a hexagonal array of negative spots,
        possibly with some stripes in place of rows of spots, and with grain boundaries that are asymptotically stable.""",
        'file_stub': "delta(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.042,
            'k': 0.059
        },
        'scaling': {
            "a": {
                'icl': 0.27,
                'im': 0.635,
                'icu': 1.0
            },
            "b": {
                'icl': 0.00,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'δ'
    },
    'delta_left': {
        'description': """
        Includes true Turing patterns and many parameter values that produce similar patterns
        through a presumably related effect. The true Turing pattern arises from a starting
        pattern that is all blue, with arbitrarily small noise or other irregularities.
        In "Turing-similar" systems, a pattern cannot arise from an all-blue starting state,
        instead they need to be "seeded" by a large amplitude disturbance (anything non-blue
        and at least as large as a typical negaton will usually suffice); and the pattern grows outward
        from the disturbance. In either case, the resulting pattern is a hexagonal array of negative spots,
        possibly with some stripes in place of rows of spots, and with grain boundaries that are asymptotically stable.""",
        'file_stub': "delta(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.030,
            'k': 0.055
        },
        'scaling': {
            "a": {
                'icl': 0.29,
                'im': 0.51,
                'icu': 0.73
            },
            "b": {
                'icl': 0.02,
                'im': 0.16,
                'icu': 0.3
            },
        },
        'symbol': 'δ'
    },
    'epsilon_right': {
        'description': """
        Spatial-temporal chaos composed mainly of spots (resembling solitons but unstable).
        Rings can grow until they contact something else; spots with less symmetry tend to live longer and
        split via mitosis. Spots continually crowd each other out; after regions are opened up by die-outs;
        other spots on the boundary of the newly opened region quickly fill it in again.

        Type alpha (α) differs from this by having no rings or symmetrical (round) spots.
        Type lambda (λ) differs from this by having no overcrowding or die-outs.
        Type zeta (ζ) differs by having spots that are somewhat less volatile,
            reproducing in a more symmetrical way and being more long-lived, though die-outs still occur.

        Pearson [1] misidentified two tests as type ε: the parameters (F, k) = (0.025, 0.0625) and
        (0.03, 0.0625) both yield type lambda (λ) behaviour.""",
        'file_stub': "epsilon(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.022,
            'k': 0.059
        },
        'scaling': {
            "a": {
                'icl': 0.2,
                'im': 0.6,
                'icu': 1
            },
            "b": {
                'icl': 0.00,
                'im': 0.21,
                'icu': 0.42
            },
        },
        'symbol': 'ε'
    },
    'epsilon_left': {
        'description': """
        Spatial-temporal chaos composed mainly of spots (resembling solitons but unstable).
        Rings can grow until they contact something else; spots with less symmetry tend to live longer and
        split via mitosis. Spots continually crowd each other out; after regions are opened up by die-outs;
        other spots on the boundary of the newly opened region quickly fill it in again.

        Type alpha (α) differs from this by having no rings or symmetrical (round) spots.
        Type lambda (λ) differs from this by having no overcrowding or die-outs.
        Type zeta (ζ) differs by having spots that are somewhat less volatile,
            reproducing in a more symmetrical way and being more long-lived, though die-outs still occur.

        Pearson [1] misidentified two tests as type ε: the parameters (F, k) = (0.025, 0.0625) and
        (0.03, 0.0625) both yield type lambda (λ) behaviour.""",
        'file_stub': "epsilon(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.018,
            'k': 0.055
        },
        'scaling': {
            "a": {
                'icl': 0.15,
                'im': 0.57,
                'icu': 1
            },
            "b": {
                'icl': 0.00,
                'im': 0.23,
                'icu': 0.46
            },
        },
        'symbol': 'ε'
    },
    'zeta_right': {
        'description': """
        Very similar to type epsilon (ε), but viable spots are generally more symmetrical and more
        stable, and the disturbances that cause spots to die are limited to a smaller percentage of the domain at any
        given time. In some ζ patterns these areas of disturbance travel slowly rather than appearing suddenly at
        random. There is also a subtle oscillation (seen in the ∂u/∂t component in the examples).

        Type epsilon (ε) has more volatile spots and constant overcrowding die-outs.
        Type lambda (λ) has no die-outs at all. """,
        'file_stub': "zeta(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.026,
            'k': 0.059
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.55,
                'icu':0.85
            },
            "b": {
                'icl': 0.0,
                'im': 0.18,
                'icu': 0.36
            },
        },
        'symbol': 'ζ'
    },
    'zeta_left': {
        'description': """
        Very similar to type epsilon (ε), but viable spots are generally more symmetrical and more
        stable, and the disturbances that cause spots to die are limited to a smaller percentage of the domain at any
        given time. In some ζ patterns these areas of disturbance travel slowly rather than appearing suddenly at
        random. There is also a subtle oscillation (seen in the ∂u/∂t component in the examples).

        Type epsilon (ε) has more volatile spots and constant overcrowding die-outs.
        Type lambda (λ) has no die-outs at all. """,
        'file_stub': "zeta(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.022,
            'k': 0.061
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.65,
                'icu': .85
            },
            "b": {
                'icl': 0.00,
                'im': 0.18,
                'icu': 0.36
            },
        },
        'symbol': 'ζ'
    },
    'eta': {
        'description': """
        Similar to type gamma (γ), but instead of all stripes there is a mixture of spots and worms.
        Any longer or more serpentine stripes will break up into short pieces. Ongoing activity includes low-level
        oscillation seen in the ∂u/∂t component, and spots occasionally splitting off and/or rejoining the worms;
        but the system always reaches a steady state if run for longer than the 200,000 tu in Pearson's experiments [1].

        Type gamma (γ) differs from this by never reaching stability.
        Type theta (θ) differs from this by having negative spots and stripes, rather than positive.
        Type kappa (κ) differs from this by being mostly stripes/worms,
            including long and/or convoluted stripes, and by having no oscillation
            in the ∂u/∂t component or by settling down to a steady state more quickly.
        Type mu (μ) differs from this by having no oscillation in the ∂u/∂t component,
            and almost no spots/solitons. """,
        'file_stub': "eta",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.034,
            'k': 0.063
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.55,
                'icu': 0.85
            },
            "b": {
                'icl': 0.0,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'η'
    },
    'theta_right': {
        'description': """
        Blue spots grow into concentric rings; stripes in isolation grow width-wise into
        parallel or crosswise stripes. Final state is mostly all stripes (sometimes with a few solitons),
        usually with branching and can be fully connected (network of loops). At lower F values there can
        be long-lasting oscillation in the ∂u/∂t component, or the system might never stop changing.

        Type gamma (γ) differs from this by having die-out events and unending oscillation in ∂u/∂t.
        Type kappa (κ) differs from this by having no stable blue state, and stripes that are less likely to join.
        Type eta (η) differs from this by having positive rather than negative spots and stripes.

        Pearson [1] misidentified one test as type θ: the parameters (F, k) = (0.03, 0.059) actually yield
        something like type gamma (γ) but with no die-out events.""",
        'file_stub': "theta(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.038,
            'k': 0.061
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.6125,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'θ'
    },
    'theta_left': {
        'description': """
        Blue spots grow into concentric rings; stripes in isolation grow width-wise into
        parallel or crosswise stripes. Final state is mostly all stripes (sometimes with a few solitons),
        usually with branching and can be fully connected (network of loops). At lower F values there can
        be long-lasting oscillation in the ∂u/∂t component, or the system might never stop changing.

        Type gamma (γ) differs from this by having die-out events and unending oscillation in ∂u/∂t.
        Type kappa (κ) differs from this by having no stable blue state, and stripes that are less likely to join.
        Type eta (η) differs from this by having positive rather than negative spots and stripes.

        Pearson [1] misidentified one test as type θ: the parameters (F, k) = (0.03, 0.059) actually yield
        something like type gamma (γ) but with no die-out events.""",
        'file_stub': "theta(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.030,
            'k': 0.057
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.6125,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.19,
                'icu': 0.38
            },
        },
        'symbol': 'θ'
    },
    'iota': {
        'description': """
        Negative spots (negatons) with molecule-like interaction; solitary negatons are not viable.
        This pattern class is closely related to type pi (π), which also supports linear and other shapes.""",
        'file_stub': "iota",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.046,
            'k': 0.0594
        },
        'scaling': {
            "a": {
                'icl': 0.28,
                'im': 0.64,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.2,
                'icu': 0.4
            },
        },
        'symbol': 'ι'
    },
    'kappa_right': {
        'description': """
        Stripes in isolation grow width-wise (usually by developing meanders to grow longer and
        more serpentine); some irregular blue shapes grow into corals. The final state is mostly all stripes,
        often with branching, and usually in multiple disjoint sets (hedgerow mazes).

        Type theta (θ) differs from this by having a stable blue state, and stripes that are more likely to join.
        Type mu (μ) differs from this by the fact that small blue spots turn into a worm,
            rather than growing out in all directions to form a ring or coral.
        Type eta (η) differs from this by having shorter stripes/worms and many solitons, and oscillation in the ∂u/∂t component.

        This evolves more slowly than the other models""",
        'file_stub': "kappa(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.058,
            'k': 0.063
        },
        'scaling': {
            "a": {
                'icl': 0.3,
                'im': 0.6,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'κ'
    },
    'kappa_left': {
        'description': """
        Stripes in isolation grow width-wise (usually by developing meanders to grow longer and
        more serpentine); some irregular blue shapes grow into corals. The final state is mostly all stripes,
        often with branching, and usually in multiple disjoint sets (hedgerow mazes).

        Type theta (θ) differs from this by having a stable blue state, and stripes that are more likely to join.
        Type mu (μ) differs from this by the fact that small blue spots turn into a worm,
            rather than growing out in all directions to form a ring or coral.
        Type eta (η) differs from this by having shorter stripes/worms and many solitons,
            and oscillation in the ∂u/∂t component.""",
        'file_stub': "kappa(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.050,
            'k': 0.063
        },
        'scaling': {
            "a": {
                'icl': 0.3,
                'im': 0.6,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.211,
                'icu': 0.422
            },
        },
        'symbol': 'κ'
    },
    'lambda_right': {
        'description': """
        Solitons that grow by mitosis (cell-division). After the space is filled,
        solitons rearrange into hexagonal grids, sometimes with grain boundaries.
        Eventually, all movement stops and the pattern is in a steady state.

        Type epsilon (ε) differs from this by having "overcrowding" and die-outs,
            and never reaching a steady state.
        Types eta (η), theta (θ), and mu (μ) differ from this by having stripes. """,
        'file_stub': "lambda(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.034,
            'k': 0.065
        },
        'scaling': {
            "a": {
                'icl': 0.2715,
                'im': 0.6357,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.1971,
                'icu': 0.39443
            },
        },
        'symbol': 'λ'
    },
    'lambda_left': {
        'description': """
        Solitons that grow by mitosis (cell-division). After the space is filled,
        solitons rearrange into hexagonal grids, sometimes with grain boundaries.
        Eventually, all movement stops and the pattern is in a steady state.

        Type epsilon (ε) differs from this by having "overcrowding" and die-outs,
            and never reaching a steady state.
        Types eta (η), theta (θ), and mu (μ) differ from this by having stripes. """,
        'file_stub': "lambda(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.026,
            'k': 0.061
        },
        'scaling': {
            "a": {
                'icl': 0.25,
                'im': 0.6025,
                'icu': 1
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'λ'
    },
    'mu_right': {
        'description': """
        Stripes that grow from each end (worms), possibly also co-existing with inert
        (non-mitotic) solitons. After the space is filled by the worms, they reorganise
        towards parallel stripes and remain disconnected from one another.

        Type eta (η) differs from this by having oscillation in the ∂u/∂t component,
            and many spots/solitons.
        Type kappa (κ) differs from this by having branching and no solitons,
            and small blue spots grow into a ring or coral rather than a worm. """,
        'file_stub': "mu(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.058,
            'k': 0.065
        },
        'scaling': {
            "a": {
                'icl': 0.2979,
                'im': 0.6489,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2204,
                'icu': 0.4409
            },
        },
        'symbol': 'μ'
    },
    'mu_left': {
        'description': """
        Stripes that grow from each end (worms), possibly also co-existing with inert
        (non-mitotic) solitons. After the space is filled by the worms, they reorganise
        towards parallel stripes and remain disconnected from one another.

        Type eta (η) differs from this by having oscillation in the ∂u/∂t component,
            and many spots/solitons.
        Type kappa (κ) differs from this by having branching and no solitons,
            and small blue spots grow into a ring or coral rather than a worm. """,
        'file_stub': "mu(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.046,
            'k': 0.065
        },
        'scaling': {
            "a": {
                'icl': 0.2872,
                'im': 0.6436,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2092,
                'icu': 0.4184
            },
        },
        'symbol': 'μ'
    },
    'nu_right': {
        'description': """
        Inert (non-mitotic) solitons. The number of solitons depends on the number and size
        of blue areas in the starting pattern. All large blue areas shrink and/or split up into solitons.
        The solitons then drift apart from each other to spread as uniformly as possible across the space,
        but this takes a period of time proportional to eKw where w is the width of the domain in lu and K
        is a constant, K≈20+1000F. In the following images (with F=0.046, k=0.067) each successive image
        represents approximately twice as much elapsed time: """,
        'file_stub': "nu(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.082,
            'k': 0.063
        },
        'scaling': {
            "a": {
                'icl': 0.2888,
                'im': 0.6444,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2116,
                'icu': 0.4233
            },
        },
        'symbol': 'ν'
    },
    'nu_left': {
        'description': """
        Inert (non-mitotic) solitons. The number of solitons depends on the number and size
        of blue areas in the starting pattern. All large blue areas shrink and/or split up into solitons.
        The solitons then drift apart from each other to spread as uniformly as possible across the space,
        but this takes a period of time proportional to eKw where w is the width of the domain in lu and K
        is a constant, K≈20+1000F. In the following images (with F=0.046, k=0.067) each successive image
        represents approximately twice as much elapsed time: """,
        'file_stub': "nu(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.054,
            'k': 0.067
        },
        'scaling': {
            "a": {
                'icl': 0.2888,
                'im': 0.6444,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2116,
                'icu': 0.4233
            },
        },
        'symbol': 'ν'
    },
    'xi_right': {
        'description': """
        Large, sustained spirals similar to the Belousov-Zhabotinsky reaction in a Petri dish.
        The seed of the spiral is an essential and sometimes rare feature, which must exist in suitable quantities
        to produce a long-lived pattern. This means that if the domain is small, the spirals will die out;
        in this case the pattern becomes uniform red state. If the domain is large enough, occasional
        irregularities usually geerate new spiral seeds and thereby maintain the population of spiral seeds.""",
        'file_stub': "xi(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.014,
            'k': 0.047
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2116,
                'icu': 0.4233
            },
        },
        'symbol': 'ξ'
    },
    'xi_left': {
        'description': """
        Large, sustained spirals similar to the Belousov-Zhabotinsky reaction in a Petri dish.
        The seed of the spiral is an essential and sometimes rare feature, which must exist in suitable quantities
        to produce a long-lived pattern. This means that if the domain is small, the spirals will die out;
        in this case the pattern becomes uniform red state. If the domain is large enough, occasional
        irregularities usually geerate new spiral seeds and thereby maintain the population of spiral seeds.""",
        'file_stub': "xi(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.010,
            'k': 0.041
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.2116,
                'icu': 0.4233
            },
        },
        'symbol': 'ξ'
    },
    'pi_left': {
        'description': """
        Type pi supports stripes, loops, and spots, all of them negative.
        These form stable localised structures, both stationary and moving,
        and localised force-like interactions whose nature (attractive or repelling)
        oscillates in sign with increasing distance and whose strength decreases exponentially with distance.
        The combination of this force interaction and motion means that you also get rotating patterns.
        Most significantly, the type pi patterns display such a great diversity that
        it is impossible to characterise the final outcome of a system just by looking at its initial state,
        except for a limited class of starting states
        (e.g. when the starting pattern consists only of lone negatons separated by large distances).""",
        'file_stub': "pi(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.062,
            'k': 0.061
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'π'
    },
    'rho_left': {
        'description': """
        Systems that are blue except for a small number of negatons evolve into
        a set of closed red "soap bubbles" bordered by stripes exhibiting "surface tension".
        These usually combine in a manner resembling real-world soap bubbles,
        with smaller bubbles shinking and vanishing whilst neighboring larger bubbles
        grow (though in limited cases at certain parameter values a negaton can persist).
        This behaviour is just like type sigma (σ) with red and blue reversed.
        However, "worm tips" (lines that end without joining with another line)
        sometimes persist or lengthen until they are constrained by the size of
        their containing bubble; in type σ such tips always shrink.

        Type ρ patterns belong to Wolfram class 2-a.""",
        'file_stub': "rho(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.09,
            'k': 0.059
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'ρ'
    },
    'rho_right': {
        'description': """
        Systems that are blue except for a small number of negatons evolve into
        a set of closed red "soap bubbles" bordered by stripes exhibiting "surface tension".
        These usually combine in a manner resembling real-world soap bubbles,
        with smaller bubbles shinking and vanishing whilst neighboring larger bubbles
        grow (though in limited cases at certain parameter values a negaton can persist).
        This behaviour is just like type sigma (σ) with red and blue reversed.
        However, "worm tips" (lines that end without joining with another line)
        sometimes persist or lengthen until they are constrained by the size of
        their containing bubble; in type σ such tips always shrink.

        Type ρ patterns belong to Wolfram class 2-a.""",
        'file_stub': "rho(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.102,
            'k': 0.055
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'ρ'
    },
    'sigma_left': {
        'description': """
        Systems that are red except for a small number of spots evolve into a set of
        closed blue "soap bubbles" bordered by red negative stripes exhibiting "surface tension".
        These bubbles usually combine in a manner resembling real-world soap bubbles,
        with smaller bubbles shinking and vanishing whilst neighboring larger bubbles
        grow (though in limited cases at certain parameter values a soliton can persist).
        This behaviour is just like type rho (ρ) with blue and red reversed.
        However, "worm tips" (lines that end without joining with another line) always shrink;
        in type ρ such tips sometimes persist or lengthen.

        Type σ patterns belong to Wolfram class 2-a.""",
        'file_stub': "sigma(left)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.09,
            'k': 0.057
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'σ'
    },
    'sigma_right': {
        'description': """
        Systems that are red except for a small number of spots evolve into a set of
        closed blue "soap bubbles" bordered by red negative stripes exhibiting "surface tension".
        These bubbles usually combine in a manner resembling real-world soap bubbles,
        with smaller bubbles shinking and vanishing whilst neighboring larger bubbles
        grow (though in limited cases at certain parameter values a soliton can persist).
        This behaviour is just like type rho (ρ) with blue and red reversed.
        However, "worm tips" (lines that end without joining with another line) always shrink;
        in type ρ such tips sometimes persist or lengthen.

        Type σ patterns belong to Wolfram class 2-a.""",
        'file_stub': "sigma(right)",
        'cmap': {'a': "seismic", 'b': 'seismic'},
        'factors': {
            'Da': 1.0,
            'Db': 0.5,
            'f': 0.11,
            'k': 0.0523
        },
        'scaling': {
            "a": {
                'icl': 0.0,
                'im': 0.5,
                'icu': 1.0
            },
            "b": {
                'icl': 0.0,
                'im': 0.22,
                'icu': 0.44
            },
        },
        'symbol': 'σ'
    },

}

PearsonGrid = [(-1.01, 1.01), (1.01, 1.01),
               (1.01, .5), (-1.01, .5),
               (-1.01, 0), (1.01, 0),
               (1.01, -.5), (-1.01, -.5),
               (-1.01, -1.01), (1.01, -1.01),
               (.5, -1.01), (.5, 1.01),
               (0, 1.01), (0, -1.01),
               (-.5, -1.01), (-.5, 1.01),
               (-1.01, 1.01)]

CurvePoints =  [[.03, .005],
                [.04, .00975],
                [.05, .019],
                [.06, .04],
                [.06275, .065],
                [.0625, .0705],
                [.06125, .08]]
PearsonScaledPoints = []

for each in CurvePoints:
    valf = ((each[0] - 0.03)/(0.07 - 0.03)-.5)*2
    valk = ((each[1] - 0.0)/(0.08 - 0.0)-.5)*2
    PearsonScaledPoints.append((valf, valk))
[y, x] = [[row[i] for row in PearsonScaledPoints] for i in range(2)]
f = interpolate.interp1d(x, y, kind='cubic')
yNew = np.arange(min(x), max(x), 0.002)
xNew = f(yNew)
PearsonCurve = list(zip(xNew, yNew))

CurvePoints2 =  [[.03, .009],
                [.035, .012],
                [.038, .014],
                [.040, .0155],
                [.044, .019],
                [.048, .023],
                [.05, .0255],
                [.054, .0315],
                [.058, .039],
                [.060, .044],
                [.062, .0545],
                [.0625, .062]]
PearsonScaledPoints2 = []

for each in CurvePoints2:
    valf = ((each[0] - 0.03)/(0.07 - 0.03)-.5)*2
    valk = ((each[1] - 0.0)/(0.08 - 0.0)-.5)*2
    PearsonScaledPoints2.append((valf, valk))
[y, x] = [[row[i] for row in PearsonScaledPoints2] for i in range(2)]
f = interpolate.interp1d(x, y, kind='quadratic')
yNew = np.arange(min(x), max(x), 0.002)
xNew = f(yNew)
PearsonCurve2 = list(zip(xNew, yNew))

# plt.plot(y, x, 'o', xNew, yNew, '-')
# plt.show()
