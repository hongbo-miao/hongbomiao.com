eps = 1 / 128;

module toroidal_propeller(
    blades = 3,                     // number of blades
    height = 6,                     // height
    blade_length = 68,              // blade length (mm)
    blade_width = 42,               // blade width (mm)
    blade_thickness = 4,            // blade thickness (mm)
    blade_hole_offset = 1.4,        // blade hole offset
    blade_twist = 15,               // blade twist angle
    blade_offset = -6,              // blade distance from propeller axis
    safe_blades_direction = "PREV", // indicates if a blade must delete itself from getting into the previous (PREV) or the next blade (NEXT).
    hub_d = 16,                     // hub diameter
    hub_screw_d = 5.5               // hub screw diameter
) {
    difference() {
        union() {
            // Hub
            cylinder(h = height, d = hub_d);

            // Propellers
            for (a = [0:blades-1]) {
                rotate([0, 0, a * (360 / blades)]) {
                    difference() {
                        translate([blade_offset, 0, 0])
                            blade(
                                height = height,
                                length = blade_length,
                                width = blade_width,
                                thickness = blade_thickness,
                                offset = blade_hole_offset,
                                twist = blade_twist
                            );

                        // Subtract what is inside other blades
                        cw_ccw_mult = (blade_twist > 0 ? -1 : 1) * (safe_blades_direction == "PREV" ? 1 : -1);
                        rotate([0, 0, cw_ccw_mult * 360 / blades])
                            translate([blade_offset, 0, -eps / 2])
                                linear_extrude(height = height + eps, twist = blade_twist)
                                    translate([blade_length / 2, 0, 0])
                                        scale([1, (blade_width - 16 * eps) / (blade_length - 16 * eps)])
                                            circle(d = blade_length - 16 * eps);
                    }
                }
            }
        }

        // Hub hole
        translate([0, 0, -eps / 2])
            cylinder(h = height + eps, d = hub_screw_d);
    }
}

module blade(height, length, width, thickness, offset, twist) {
    linear_extrude(height = height, twist = twist, convexity = 2)
        difference() {
            translate([length / 2, 0, 0])
                scale([1, width / length])
                    circle(d = length);

            translate([length / 2 + offset, 0, 0])
                scale([1, (width - thickness) / (length - thickness)])
                    circle(d = length - thickness);
        }
}
