SetFactory("Built-in");
Mesh.MshFileVersion=2.2;                // force .msh version 2

radius = 1;    // radius [=] m
length = 10;  // height [=] m

n_layers_r     =  20;  // Number of elements in r for the pie slices
n_layers_theta =  10;  // Number of elements in theta for each slide (1/6 of the circle)
n_layers_z     =  50;  // Number of elements in z for the inlet straight
n_layers_hex   =  n_layers_theta;  // Number of elements to have on inner surfaces of hexagon

divisor = 1.5;

// Center point
Point(0) = {0, 0, 0}; 

// Add points for outer circle
// Points 1-6 starting with (x, 0) and going clockwise
For i In {0:5}
    Point(i+1) = {radius * Cos(i*Pi/3), radius * Sin(i*Pi/3), 0};
EndFor

// I don't know what this is for
p1 = newp;

// Add points for inner hexagon
// Points 8-13 starting with (x, 0) and going clockwise
For i In {0:5}
    Point(p1+i+1) = {radius * Cos(i*Pi/3)/divisor, radius * Sin(i*Pi/3)/divisor, 0};
EndFor

// Lines to build the outside of the hexagon, starting at (x, 0) and going COUNTER-clockwise
Line(1) = {11, 10};
Line(2) = {10, 9};
Line(3) = {9, 8};
Line(4) = {8, 13};
Line(5) = {13, 12};
Line(6) = {12, 11};

// Lines to build the inside of the hexagon
Line(7) = {11, 0};
Line(8) = {0, 9};
Line(9) = {0, 13};

// The circle proper
Circle(10) = {4, 0, 5};
Circle(11) = {5, 0, 6};
Circle(12) = {6, 0, 1};
Circle(13) = {1, 0, 2};
Circle(14) = {2, 0, 3};
Circle(15) = {3, 0, 4};

// Lines connecting hexagon to circle
Line(16) = {5, 12};
Line(17) = {6, 13};
Line(18) = {1, 8};
Line(19) = {2, 9};
Line(20) = {3, 10};
Line(21) = {4, 11};

// Surfaces for the circle outside the hexagon
Curve Loop(1) = {10, 16, 6, -21};
Plane Surface(1) = {1};
Curve Loop(2) = {16, -5, -17, -11};
Plane Surface(2) = {2};
Curve Loop(3) = {17, -4, -18, -12};
Plane Surface(3) = {3};
Curve Loop(4) = {18, -3, -19, -13};
Plane Surface(4) = {4};
Curve Loop(5) = {19, -2, -20, -14};
Plane Surface(5) = {5};
Curve Loop(6) = {20, -1, -21, -15};
Plane Surface(6) = {6};

// Surfaces inside the hexagon
Curve Loop(7) = {1, 2, -8, -7};
Plane Surface(7) = {7};
Curve Loop(8) = {7, 9, 5, 6};
Plane Surface(8) = {8};
Curve Loop(9) = {4, -9, 8, 3};
Plane Surface(9) = {9};

// Outer faces of the hexagon
Transfinite Curve {1:6} = n_layers_theta+1 Using Progression 1;
// Inner faces of the hexagon
Transfinite Curve {7:9} = n_layers_hex+1 Using Progression 1;
// Perimiter of the circle
Transfinite Curve {10:15} = n_layers_theta+1 Using Progression 1;
// Lines connecting hexagon and circle
Transfinite Curve {16:21} = n_layers_r+1 Using Progression 1.2;

// Add transfinite surfaces BUT NOT ON THE HEXAGONS
For i In {1:6}
    Transfinite Surface {i};
EndFor

// Combine triangular-prisms into rhombic-prisms (pentahedrons to hexahedrons)
For i In {1:9}
    Recombine Surface {i};
EndFor

Extrude {0, 0, length} { Surface{1:9}; Layers{n_layers_z}; Recombine; }
//+
Physical Surface("inlet") = {3, 4, 5, 6, 1, 2, 9, 8, 7};
//+
Physical Surface("outlet") = {109, 131, 153, 43, 65, 87, 175, 197, 219};
//+
Physical Surface("wall") = {86, 108, 64, 30, 152, 130};
//+
Physical Volume("inside") = {2, 3, 1, 8, 9, 7, 4, 6, 5};
