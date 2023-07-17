use <utils/toroidal_propeller.scad>

$fn = 100;
toroidal_propeller(
    blades = 3,
    height = 6,
    blade_length = 68,              // mm
    blade_width = 42,               // mm
    blade_thickness = 4,            // mm
    blade_hole_offset = 1.4,
    blade_twist = 15,
    blade_offset = -6,
    safe_blades_direction = "PREV", // NEXT
    hub_d = 16,
    hub_screw_d = 5.5
);
