SetFactory("OpenCASCADE");
// force .msh version 2
Mesh.MshFileVersion=2.2;

Rectangle(1) = {0, 0, 0, 1, 1, 0};
Transfinite Curve {1, 4, 3, 2} = 70 Using Progression 1;
Transfinite Surface {1};
Mesh 2;//+
Physical Surface("inside", 5) = {1};
//+
Physical Curve("wall_fixed", 6) = {4, 1, 2};
//+
Physical Curve("wall_moving", 7) = {3};
