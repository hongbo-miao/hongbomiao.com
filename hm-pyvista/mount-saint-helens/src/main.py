import pyvista as pv


def main() -> None:
    mesh = pv.examples.download_st_helens().warp_by_scalar()
    pv.set_plot_theme("document")
    p = pv.Plotter()
    p.add_mesh(mesh)
    p.add_bounding_box()
    p.show()


if __name__ == "__main__":
    main()
