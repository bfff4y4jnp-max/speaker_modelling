"""Write demo STEP files (stand-ins for your Fusion/FreeCAD/SolidWorks exports).

Only the solid parts (panels, port tube) are exported; pistons are placed in
project.yml. Units: mm, as most CAD tools export.
"""
import os
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))


def box(name, W, H, D, t, holes, tubes=()):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    o = occ.addBox(0, 0, 0, W, H, D)
    i = occ.addBox(t, t, t, W - 2 * t, H - 2 * t, D - 2 * t)
    out, _ = occ.cut([(3, o)], [(3, i)])
    cyl = [(3, occ.addCylinder(x - 50, y, z, 100, 0, 0, d / 2)) for x, y, z, d in holes]   # holes through the x=W baffle
    out, _ = occ.cut(out, cyl)
    for x, y, z, d, L, w in tubes:                                                        # port tubes along -x
        to = occ.addCylinder(x, y, z, -L, 0, 0, d / 2 + w)
        ti = occ.addCylinder(x, y, z, -L, 0, 0, d / 2)
        tt, _ = occ.cut([(3, to)], [(3, ti)])
        out, _ = occ.fuse(out, tt)
    occ.synchronize()
    gmsh.write(os.path.join(HERE, "cad", name + ".step"))
    gmsh.finalize()


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "cad"), exist_ok=True)
    # vented sub: 500 x 450 x 550 mm, 18 mm ply, 12" woofer, 100 mm port on the front (x = 500 mm) baffle
    box("sub", 500, 450, 550, 18, holes=[(500, 225, 330, 280), (500, 225, 90, 100)],
        tubes=[(500, 225, 90, 100, 200, 6)])
    # sealed top: 250 x 250 x 350 mm, 6.5" driver
    box("top", 250, 250, 350, 15, holes=[(250, 125, 220, 150)])
