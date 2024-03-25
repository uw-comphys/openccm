SetFactory("OpenCASCADE");
Mesh.MshFileVersion=2.2;


Rectangle(1) = {0, 0, 0, 1, 1, 0};

// Top and bottom
Transfinite Curve {4, 2} = 1 Using Progression 1;
// Left and right
Transfinite Curve {1, 3} = 500 Using Progression 1;


Physical Curve("left", 5) 	= {4};
Physical Curve("right", 6) 	= {2};
Physical Curve("top", 7) 	= {3};
Physical Curve("bottom", 8) = {1};
Physical Surface("inside") 	= {1};

Transfinite Surface {1};


Recombine Surface {1};
Mesh 2;