include <gears.scad>;

/* [Print option] */

// Circle segments 64 for draft/testing, 1024 for 3d print
rot_steps = 128;  // [64, 128, 256, 512, 1024]

// Toggle creation of inner ring of the ball bearing.
create_inner_bearing=false;

// Toggle creation of outer ring of the ball bearing (top).
create_outer_bearing_top=false;

// Toggle creation of outer ring of the ball bearing (bottom).
create_outer_bearing_bottom=false;

// Toggle creation of the base
create_base=false;

// Toggle creation of the gears
create_gears=true;

// Toggle creation of flat top platform
create_platform=false;


/* [Structure] */

// Ring gear teath. Use a highly divisable number, so the gearbox is happy. Note: this setting also changes the size of the ring.
ring_teeth = 54; 

// The gear box speeds (should devide into ring_teeth). eg: 27/54 = 1/2, 18/54 = 1/3; 9/54 = 1/6
drive_gear_teeth_sizes = [18, 9];

// Use alternate servo placement scheme.
linear_placement=false;

// The size of the turn table (mm). If this is not more than 20mm greater than outer_diameter; things break.
large_bearing_outer_dia=130;

// Size of inner bearing's outer diameter (the name could have been better) (mm).
outer_diameter = 100; 

// The width of the top table, to which the servos and bearings are attached (mm). Don't make it too thin.
plate_thickness = 3;

// Number of holes to attach bearings
num_mount_holes = 8; // [4:2:16]

// Distance between the iner bearing ring and the table it sits on (mm).
bearing_gap = 1;

// Height of the motor enclosure (mm).
enclosure_height=50;

// size of pillars that hold up the table, and provide rigidity to everything.
enclosure_pillar_size = 8;

// Thickness of material in the lower enclosure, note this can be thin as support pillars are present.
enclosure_thickness = 2.5;

// The size of hole that the wires to the table top can pass through. 
inner_tube_dia = 20;


/* [Gears (advanced)] */
// (advanced) Increase to strenghten gears, decrease to reduce noise.
pressure_angle=20; // [17.5:0.5:22.5]
// (advanced) angle of helix
helix_angle=20; // [0:5:45]
// (advanced) ratio diameter by the number of teeth
gear_module = 1.5; // [0.5:0.5:2.5]
// gear height (mm)
gear_width = 4; // [2:0.1:12]

/* [Screws] */
// use 3.2 for m3 bolts
bolt_hole_size=3.2; // [2:0.1:5]
// use 2.5 for m3 bolts
bolt_tap_size=2.5; // [1.5:0.1:4.5]

/* [Bearing Balls] */
// Bearing ball size (mm)
bearing_ball_dia=6;

// The holes to bolt down a bearing are mesured from the centre of the hole, to the outer edge off bearing (mm)
mount_hole_recess_from_bearing_edge=8;


bearing_channel_size=bearing_ball_dia/2;
bearing_dia =  outer_diameter + bearing_ball_dia - bearing_channel_size;
large_bearing_inner_dia = outer_diameter + 2*bearing_ball_dia - bearing_channel_size*2;
bearing_height = bearing_ball_dia*2;


/* [Servos] */
// Dimension of the grippy part of the servo head
servo_head_dia = 5.75;
// Depth of the grippy part of the servo head, (how far too collar it)
servo_head_height = 3.5;

// Dimension of bolt (mm) to fasten to the servo head (3.25 for m3 bolt)
servo_scew_hole = 3.25;

// Pocket dia for head of bolt (mm) that fastens to the servo head, 6mm is safe for m3 bolt
servo_scew_head_dia = 6;
// Pocket depth for head of bolt (mm) that fastens to the servo head, 2.2 for m3 bolt (acording to my calipers).
servo_scew_head_depth = 2.2;

// Dimension of servo housing box (excluding bracket) (mm)
servo_size_x = 40.5;
// Dimension of servo housing box (excluding bracket) (mm)
servo_size_y = 21;

// Size of bracket protrusion (mm).
servo_bracket_size = (55-servo_size_x)/2;

// Distance from top of bracket to top of servo head (mm).
servo_bracket_to_top = 42.15 - 29.1;

// Distance from top of bracket to the flat of the servo box (mm).
servo_bracket_to_lid = 37.1 - 29.1;

// Thickness of servo bracket (mm).
servo_bracket_thickness = 29.1 - 26.4;

// Thickness of servo bracket support (mm)
servo_gusset_width = 2.2;

// Center of shaft to edge of servo housing box (excluding bracket) (mm)
servo_shaft_offset = 9.73;

// Distance between center of mounting holes (mm)
servo_hole_space_x = 48.5;

// Distance between center of mounting holes (mm)
servo_hole_space_y = 10;

// How big are the holes in the bracket (better to be slightly too small) (mm). 
servo_hole_dia = 4.7;

// how big are your servo mounting bolts (2.0 will let you mount with the self tappers often supplied with the servo) (mm).
servo_bolt_dia = 2;

servo_lid_to_top = servo_bracket_to_top - servo_bracket_to_lid;

// gear calculation steps
function pitch_diameter(teeth) = gear_module * teeth;
function tip_clearance(teeth) = (teeth <3)? 0 : gear_module/6;
function root_diameter(teeth) =
    pitch_diameter(teeth) -
    (2*(gear_module + tip_clearance(teeth)));
function total_diameter(teeth) = 
    pitch_diameter(teeth) + 
    (pitch_diameter(teeth) - root_diameter(teeth));

//padding to the ring gear, to make it hit the outer diameter
rim_size = (outer_diameter - total_diameter(ring_teeth)) / 2;
//operating_distance = pitch_diameter(ring_teeth)/2 - pitch_diameter(drive_gear_teeth)/2;



// creates a tube of length h, centered in x/y, but not z.
module tube(h, inner_d, outer_d)
{
    translate([0,0,h/2])
        difference()
        {
            // outer cylinder
            cylinder(h=h, d=outer_d, center=true, $fn=rot_steps);
            if (inner_d > 0) 
                translate([0,0,-1])
                    // inner cylinder
                    cylinder(h=h+4, d=inner_d, center=true, $fn=rot_steps);
        }    
}

module bearing_recess()
{
    rotate_extrude(convexity = 10, $fn=rot_steps)
    translate([bearing_dia/2, 0, 0]) 
    circle(r = bearing_ball_dia/2, $fn=rot_steps/2);
}

module bearing_balls()
{
    n=64;
    for (i = [0 : n]) {
        
        rotate( i * (360/n), [0, 0, 1])
            translate([bearing_dia/2, 0, 0])     
                sphere(r = bearing_ball_dia/2, $fn=16);
    }
}

module mount_holes(n, h, dia, bolt_dia, with_head=false)
{
    for (i = [0 : n]) {
        rotate( i * (360/n), [0, 0, 1])
        translate([dia/2, 0, 0])     
        {
            if (with_head) {
                union() {
                    cylinder(h=h, r=bolt_dia/2);
                    translate([0, 0, h-2])     
                        cylinder(h=2, r=bolt_dia);
                }
            }
            else {
                cylinder(h=h, r=bolt_dia/2);
            }
        }
    }
}


module inner_bearing_with_ring_gear()
{
    difference() {
        union() {
            tube(bearing_height, outer_diameter-14, outer_diameter); 
            herringbone_ring_gear (
                    modul=gear_module, 
                    tooth_number=ring_teeth, 
                    width=gear_width, 
                    rim_width=rim_size, 
                    pressure_angle=pressure_angle, 
                    helix_angle=helix_angle);
        }
        // allow bearing size again amount of material
        translate([0,0,bearing_ball_dia])
            bearing_recess();
    }
}

module servo_gear(drive_gear_teeth, boss_len = 4, servo_head_to_base_of_gear = 0)
{
    rotate([0,0,360/drive_gear_teeth/2])
    {
        difference()
        {
           union()
           {
                
                d1 = servo_head_dia + 1;
                d2 = servo_head_dia + 3;
                translate([0,0,-boss_len])
                    tube(boss_len+gear_width, servo_head_dia, d2); // collar
                translate([0,0,-servo_head_to_base_of_gear])
                    tube(gear_width+servo_head_to_base_of_gear, servo_scew_hole, servo_head_dia); // through hole and seet for servo head
                herringbone_gear (
                            modul=gear_module, 
                            tooth_number=drive_gear_teeth, 
                            width=gear_width, 
                            bore=d1,
                            pressure_angle=pressure_angle, 
                            helix_angle=helix_angle);
           }
           translate([0,0,gear_width])
           translate([0,0,-servo_scew_head_depth+0.01])
           cylinder(h=servo_scew_head_depth, d=servo_scew_head_dia); // let a m3 head sit inside
        }
    }
}

module outer_bearing(top)
{
    difference() {
        tube(bearing_height, 
             large_bearing_inner_dia, 
             large_bearing_outer_dia);
        // todo: sholder
        
        // allow bearing size again amount of material
        translate([0,0,bearing_ball_dia])
            bearing_recess();
        if(top) {
            // +10 height to remove slivers
            translate([0,0,-10])
            cylinder(h=bearing_height/2+10, d=large_bearing_outer_dia+1);
        }
        else{
            // +10 height to remove slivers
            translate([0,0,bearing_height/2])
            cylinder(h=bearing_height/2+10, d=large_bearing_outer_dia+1);
        }
    }
}


module servo_pilar(extra=0)
{
    difference() {
        cube([servo_bracket_size, servo_size_y, servo_bracket_to_lid+extra
        ]);
        translate([0,servo_size_y/2,0])
        translate([0,-servo_gusset_width/2,0])
            cube([servo_bracket_size, servo_gusset_width, 1]);
        }
}

module servo_holes(height, dia)
{
    translate([0,servo_size_y/2,-servo_bracket_thickness])
    {
        for(t=[1,-1])
        translate([0,t*(servo_hole_space_y/2),0])
        {
            servo_to_hole = (servo_hole_space_x - servo_size_x) / 2;
            translate([servo_bracket_size - servo_to_hole, 0, 0])
            tube(height, 0, dia); 
            
            translate([servo_bracket_size +servo_size_x+servo_to_hole, 0, 0])
            tube(height, 0, dia); 
        }
    }
}

module servo_bracket(extra=0)
{
    translate([-(servo_bracket_size+servo_shaft_offset),
                -servo_size_y/2,
                -servo_bracket_to_top])
    {
        difference()
        {
            union() 
            {
                servo_pilar(extra);
                translate([servo_size_x+servo_bracket_size,0,0])
                servo_pilar(extra);
                
                // boss through servo mount hole
                translate([0,0,0.1]) //push it down so its not higher than the servo
                    servo_holes(servo_bracket_thickness, servo_hole_dia);
            }
            servo_holes(10, servo_bolt_dia);
        }
        
        translate([servo_bracket_size,0,-37.1+servo_bracket_to_lid])
        %cube([servo_size_x, servo_size_y, 37.1]);
        translate([0,0,-servo_bracket_thickness])
        %cube([servo_size_x+2*servo_bracket_size, servo_size_y, servo_bracket_thickness]);
    }
    
    translate([0,0,-servo_bracket_to_top/2])
        %cylinder(h=servo_bracket_to_top, d=5.75, center=true, $fn=8);
}

module platform(platform_width = 3)
{
    // This is a trivial platform.
    // You probably have something you wish to spin.
    // This function is a starting point for that.
    
    color([0,0.5,1, 0.9])
    translate([0,0,bearing_height+1])
        difference()
        {
            
            union() {
                //platform
                cylinder(h=platform_width, d=large_bearing_outer_dia, $fn=rot_steps);
                
                // a protruded area, so the platform does not sit on the outer bearing.
                translate([0, 0, -1])
                    cylinder(h=platform_width, d=outer_diameter, $fn=rot_steps);
            }
            
            // wire hole
            translate([0,0,-2])
                cylinder(h=platform_width+4, d=inner_tube_dia, $fn=rot_steps/2);
                
            // mounting holes to match the inner bearing ring.
            translate([0,0,-1.1])  // 0.1mm either side, to clean up slivers.
                mount_holes(num_mount_holes, platform_width+0.2+1, outer_diameter-mount_hole_recess_from_bearing_edge, bolt_tap_size, with_head=false);
        }
}

// ---------------------------------------------------------------------
// Render and assemble
// ---------------------------------------------------------------------

positions = len(drive_gear_teeth_sizes);



// the top plate (to which everything attaches)
if(create_base)
{
    union()
    {
        translate([0, 0, -plate_thickness])
        difference() {
            // the plate
            union()
            {
                tube(plate_thickness, inner_tube_dia, large_bearing_outer_dia);
                
                // A support collar under the plate, to make everything more rigid. 
                // Also gives some purchase for the bolts
                support_h = 5;
                support_w = mount_hole_recess_from_bearing_edge + bolt_hole_size ;
                translate([0,0,-support_h]) {
                    tube(support_h, large_bearing_outer_dia-support_w, large_bearing_outer_dia);
                
                    // make the support collar thicker around the mount holes.
                    mount_holes(num_mount_holes, support_h, large_bearing_outer_dia-mount_hole_recess_from_bearing_edge, mount_hole_recess_from_bearing_edge, with_head=false);
                }
            }
            
            mount_hole_spacing = 5;
            max_dia = root_diameter(ring_teeth);
            for(i=[0:floor((max_dia/2)/mount_hole_spacing)]) {
                dia = mount_hole_spacing*2 * i;
                if (dia > inner_tube_dia)
                    mount_holes(4, plate_thickness+5, dia, bolt_hole_size);
            }
            
            for (i = [0:positions-1]) {
                drive_gear_teeth = drive_gear_teeth_sizes[i];
                operating_distance = pitch_diameter(ring_teeth)/2 - pitch_diameter(drive_gear_teeth)/2;
                
                rotate( i * (360/positions), [0, 0, 1])
                translate([0, operating_distance, 0])
                translate([0, 0, -0.5])
                tube(4, 0, 15);
            }
            // holes to attach upper bearing (just made realy long to go through material)
            translate([0,0,-50])
                mount_holes(num_mount_holes, 100, large_bearing_outer_dia-mount_hole_recess_from_bearing_edge, bolt_tap_size, with_head=false);
        }
        
        
        // servo brackets
        for (i = [0:positions-1]) {
            drive_gear_teeth = drive_gear_teeth_sizes[i];
            operating_distance = pitch_diameter(ring_teeth)/2 - pitch_diameter(drive_gear_teeth)/2;
            
            rotate( i * (360/positions), [0, 0, 1])
            translate([0, operating_distance, 0])
            rotate(linear_placement ? (-i * (360/positions)+90) : 0, [0, 0, 1])
            
                // The servo bracket goes to the top of the surface to fill in mounting holes it covers up (keeping to safe for 3d printing).
                servo_bracket(servo_lid_to_top);
        }
        
        // enclosure
        enclosure_mount_hole_dia = large_bearing_outer_dia- (enclosure_pillar_size/2);
        echo("Circle of 8 enclosure holes is diameter ", enclosure_mount_hole_dia);
        translate([0,0,-enclosure_height])
        {
            
            
            rotate(-(360/32), [0, 0, 1])
            difference()
            {
                union() {
                    tube(enclosure_height, large_bearing_outer_dia-enclosure_thickness, large_bearing_outer_dia);
                    intersection() {
                        // remove part of pillar that goes outside enclosure
                        tube(enclosure_height, 0, large_bearing_outer_dia);  
                        // pilars
                        mount_holes(num_mount_holes, enclosure_height, large_bearing_outer_dia, enclosure_pillar_size);
                    }
                }
             
            // holes
            translate([0,0,-1])
                mount_holes(num_mount_holes, enclosure_height, enclosure_mount_hole_dia, bolt_tap_size);
            }
        }

    }
}



// gears
if(create_gears)
{
    color([1,0,1]){
        for (i = [0:positions-1]) {
            drive_gear_teeth = drive_gear_teeth_sizes[i];
            operating_distance = pitch_diameter(ring_teeth)/2 - pitch_diameter(drive_gear_teeth)/2;
            
            rotate( i * (360/positions), [0, 0, 1])
            translate([0, operating_distance, bearing_gap])
                servo_gear(drive_gear_teeth, bearing_gap+servo_head_height, bearing_gap);
        }
    }
}


//bearing
translate([0,0,bearing_gap])
{    
    if(create_inner_bearing)
    {
        color([0,1,1])
        difference()
        {
            inner_bearing_with_ring_gear();
            mount_holes(num_mount_holes, bearing_height, outer_diameter-mount_hole_recess_from_bearing_edge, bolt_tap_size);
        }
        
        // allow bearing size again amount of material
        translate([0,0,bearing_ball_dia])
            %bearing_balls();
    }
    
    
    // platform
    if(create_platform) {
        platform(platform_width = 3);
    }


    if(create_outer_bearing_top || create_outer_bearing_bottom)
    {
        color([0,1,1])
        difference()
        {
            union() {
                if(create_outer_bearing_top) {
                    outer_bearing(top=true);
                }
                
                if(create_outer_bearing_bottom) {
                    outer_bearing(top=false);
                    // mounting rim
                translate([0,0,-bearing_gap])
                    tube(bearing_gap, large_bearing_inner_dia, large_bearing_outer_dia);
                }
                
            }
            mhs = bearing_height+bearing_gap;
            translate([0,0,-bearing_gap])
                mount_holes(num_mount_holes, mhs, large_bearing_outer_dia-mount_hole_recess_from_bearing_edge, bolt_hole_size, with_head=true);
        }
    }
    
}
