# https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/01%20-%20AeroSandbox%203D%20Aerodynamics%20Tools/01%20-%20Vortex%20Lattice%20Method/01%20-%20Vortex%20Lattice%20Method.ipynb

import logging

import aerosandbox as asb
import aerosandbox.numpy as np
import aerosandbox.tools.pretty_plots as p
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def main() -> None:
    wing_airfoil = asb.Airfoil("sd7037")
    tail_airfoil = asb.Airfoil("naca0010")

    # Define the 3D geometry you want to analyze/optimize
    # All distances are in meters and all angles are in degrees
    airplane = asb.Airplane(
        name="Glider",
        xyz_ref=[0, 0, 0],  # CG location
        wings=[
            asb.Wing(
                name="Main Wing",
                # Should this wing be mirrored across the XZ plane
                symmetric=True,
                # The wing's cross ("X") sections
                xsecs=[
                    # Root
                    asb.WingXSec(
                        # Coordinates of the XSec's leading edge, relative to the wing's leading edge
                        xyz_le=[0, 0, 0],
                        chord=0.18,
                        twist=2,  # degrees
                        # Airfoils are blended between a given XSec and the next one
                        airfoil=wing_airfoil,
                    ),
                    # Mid
                    asb.WingXSec(
                        xyz_le=[0.01, 0.5, 0],
                        chord=0.16,
                        twist=0,
                        airfoil=wing_airfoil,
                    ),
                    # Tip
                    asb.WingXSec(
                        xyz_le=[0.08, 1, 0.1],
                        chord=0.08,
                        twist=-2,
                        airfoil=wing_airfoil,
                    ),
                ],
            ),
            asb.Wing(
                name="Horizontal Stabilizer",
                symmetric=True,
                xsecs=[
                    # Root
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=-10,
                        airfoil=tail_airfoil,
                    ),
                    # Tip
                    asb.WingXSec(
                        xyz_le=[0.02, 0.17, 0],
                        chord=0.08,
                        twist=-10,
                        airfoil=tail_airfoil,
                    ),
                ],
            ).translate([0.6, 0, 0.06]),
            asb.Wing(
                name="Vertical Stabilizer",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=0.1,
                        twist=0,
                        airfoil=tail_airfoil,
                    ),
                    asb.WingXSec(
                        xyz_le=[0.04, 0, 0.15],
                        chord=0.06,
                        twist=0,
                        airfoil=tail_airfoil,
                    ),
                ],
            ).translate([0.6, 0, 0.07]),
        ],
        fuselages=[
            asb.Fuselage(
                name="Fuselage",
                xsecs=[
                    asb.FuselageXSec(
                        xyz_c=[0.8 * xi - 0.1, 0, 0.1 * xi - 0.03],
                        radius=0.6 * asb.Airfoil("dae51").local_thickness(x_over_c=xi),
                    )
                    for xi in np.cosspace(0, 1, 30)
                ],
            ),
        ],
    )
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            velocity=25,  # m/s
            alpha=5,  # degree
        ),
    )
    aero = vlm.run()
    for k, v in aero.items():
        logger.info(f"{k.rjust(4)} : {v}")
    vlm.draw(show_kwargs=dict(jupyter_backend="static"))

    # Operating Point Optimization
    opti = asb.Opti()
    alpha = opti.variable(init_guess=5)
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=asb.OperatingPoint(velocity=25, alpha=alpha),
        align_trailing_vortices_with_wind=False,
    )
    aero = vlm.run()
    l_over_d = aero["CL"] / aero["CD"]
    opti.minimize(-l_over_d)
    sol = opti.solve()
    best_alpha = sol(alpha)
    logger.info(f"Alpha for max L/D: {best_alpha:.3f} deg")

    # Aerodynamic Shape Optimization
    opti = asb.Opti()
    chord_section_number = 16
    section_y = np.sinspace(0, 1, chord_section_number, reverse_spacing=True)
    chords = opti.variable(init_guess=np.ones(chord_section_number))
    wing = asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[-0.25 * chords[i], section_y[i], 0], chord=chords[i])
            for i in range(chord_section_number)
        ],
    )
    airplane = asb.Airplane(wings=[wing])
    opti.subject_to([chords > 0, wing.area() == 0.25])
    opti.subject_to(np.diff(chords) <= 0)
    alpha = opti.variable(init_guess=5, lower_bound=0, upper_bound=30)
    op_point = asb.OperatingPoint(
        velocity=1,
        alpha=alpha,
    )
    vlm = asb.VortexLatticeMethod(
        airplane=airplane,
        op_point=op_point,
        spanwise_resolution=1,
        chordwise_resolution=8,
    )
    aero = vlm.run()
    opti.subject_to(aero["CL"] == 1)
    opti.minimize(aero["CD"])
    sol = opti.solve()
    vlm = sol(vlm)
    vlm.draw(show_kwargs=dict(jupyter_backend="static"))

    # Compare our optimized solution with known analytic solution (an elliptical lift distribution)
    plt.plot(
        section_y,
        sol(chords),
        ".-",
        label="AeroSandbox VLM Result",
        zorder=4,
    )
    y_plot = np.linspace(0, 1, 500)
    plt.plot(
        y_plot,
        (1 - y_plot**2) ** 0.5 * 4 / np.pi * 0.125,
        label="Elliptic Distribution",
    )
    p.show_plot(
        "AeroSandbox Drag Optimization using VortexLatticeMethod",
        "Span [m]",
        "Chord [m]",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
