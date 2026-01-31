SetFactory("OpenCASCADE");
// force .msh version 2
Mesh.MshFileVersion=2.2;

// Variables
  // Radius of the outer circle
  r   = 1.0;

  // Inlet length
  len_in  = 1.0;
  // Outlet length
  len_out = 5.0;
  // Total length of the pipe
  len_total = len_in + 2*r + len_out;

  // Side length of inset rectangle for semi-circle
  len_square = (2*r) * 0.3;

  // Height of the pipe
  height_pipe = 0.5;

  // Number of elments for different regions
    n_elements_width      = 12;
    n_elements_inlet      = 15;

    n_elements_outlet     = Round(len_out * n_elements_inlet / len_in); 
    n_elements_perimiter  = Round(r*Pi/4 * n_elements_inlet / len_in); 
    n_elements_radial     = Round(r * n_elements_inlet / len_in); 

    n_layers              = 20;

  // Progressions for the transfinite curves
    // Perpendicular to flow in pipe (Capture boundary layer)
    p_perp = 1.2;

// Create the half-circle
  // Points needed to make the half-circle
    // Center
      Point(1) = {  
        len_in + r,
        0,
        0,
        1.0
      };
    // Left
      Point(2) = {
        len_in,
        0,
        0,
        1.0
      };
    // 1/4rd
      Point(3) = {
        len_in + r*(1+Cos(Pi * 5/4)),
        -r * Sin(Pi/4),
        0,
        1.0
      };
    // 3/4rds
      Point(4) = {
        len_in + r*(1 + Cos(Pi * 7/4)),
        -r * Sin(Pi/4),
        0,
        1.0
      };
    // Right
      Point(5) = {
        len_in + 2*r,
        0,
        0,
        1.0
      };
  // Create arcs
    Circle(1) = {2, 1, 3};
    Circle(2) = {3, 1, 4};
    Circle(3) = {4, 1, 5};

  // Create the square needed to fill the semi-circle
    // Only create the bottom half of the square, since that's all we need.
      Rectangle(1) = {
        len_in + r - len_square/2, -len_square/2, 0,   // Bottom left corner
        len_square,                len_square/2,  0    // Size (dx, dy, dz)
      };

  // Connect up the inside of the semi-circle
    Line(8)  = {2, 9};
    Line(9)  = {3, 6};
    Line(10) = {4, 7};
    Line(11) = {5, 8};

  // Create the loops and surfaces
    Curve Loop(2)    = {1,8,7,9};
    Plane Surface(2) = {2};

    Curve Loop(4)    = {2,9,4,10};
    Plane Surface(3) = {4};

    Curve Loop(6)    = {3,10,5,11};
    Plane Surface(4) = {6};

// Create the layers above the semi-circle
  // Above the left side
    Point(10) = {len_in,                      height_pipe/2,  0, 1.0};
    Point(11) = {len_in + r - len_square/2,   height_pipe/2,  0, 1.0};
    Point(12) = {len_in,                      height_pipe,    0, 1.0};
    Point(13) = {len_in + r - len_square/2,   height_pipe,    0, 1.0};
    
    // Bottom
      Line(12) = {2, 10};
      Line(13) = {10, 11};
      Line(14) = {11, 9};

    // Top
      Line(15) = {10, 12};
      Line(16) = {12, 13};
      Line(17) = {13, 11};

    // Loops and surfaces
      Curve Loop(8)    = {12, 13, 14, -8};
      Plane Surface(5) = {8};

      Curve Loop(9)    = {13, -17, -16, -15};
      Plane Surface(6) = {9};

  // Above the middle
    Point(14) = {len_in + r + len_square/2, height_pipe/2,  0, 1.0};
    Point(15) = {len_in + r + len_square/2, height_pipe,    0, 1.0};

    // Bottom
      Line(18) = {11, 14};
      Line(19) = {14, 8};

    // Top
      Line(20) = {13, 15};
      Line(21) = {15, 14};

    // Loops and surfaces
      Curve Loop(10) = {6, -14, 18, 19};
      Plane Surface(7) = {10};
      
      Curve Loop(11) = {18, -21, -20, 17};
      Plane Surface(8) = {11};

  // Above the right side
    Point(16) = {len_in + 2*r, height_pipe/2,  0, 1.0};
    Point(17) = {len_in + 2*r, height_pipe,    0, 1.0};

    // Bottom
      Line(22) = {14, 16};
      Line(23) = {16, 5};

    // Top
      Line(24) = {15, 17};
      Line(25) = {17, 16};

    // Loops and surfaces
      Curve Loop(12) = {21, 22, -25, -24};
      Plane Surface(9) = {12};

      Curve Loop(13) = {19, -11, -23, -22};
      Plane Surface(10) = {13};

// Create the layers for the entry region
  Point(18) = {0, 0,              0, 1.0};
  Point(19) = {0, height_pipe/2,  0, 1.0};
  Point(20) = {0, height_pipe,    0, 1.0};

  // Bottom
    Line(26) = {2, 18};
    Line(27) = {18, 19};
    Line(28) = {19, 10};

  // Top
    Line(29) = {19, 20};
    Line(30) = {20, 12};

  // Loops and surfaces
    Curve Loop(14) = {27, 28, -12, 26};
    Plane Surface(11) = {14};

    Curve Loop(15) = {29, 30, -15, -28};
    Plane Surface(12) = {15};

// Create the layers for the exit region
  Point(21) = {len_in + 2*r + len_out, 0,             0, 1.0};
  Point(22) = {len_in + 2*r + len_out, height_pipe/2, 0, 1.0};
  Point(23) = {len_in + 2*r + len_out, height_pipe,   0, 1.0};

  // Bottom
      Line(31) = {16, 22};
      Line(32) = {22, 21};
      Line(33) = {21, 5};

  // Top
    Line(34) = {17, 23};
    Line(35) = {23, 22};

  // Loops and surfaces
    Curve Loop(16) = {23, -33, -32, -31};
    Plane Surface(13) = {16};

    Curve Loop(17) = {25, 31, -35, -34};
    Plane Surface(14) = {17};

// Add Transfinite Curves
    // Pipe Cross-section
      // Bottom
        Transfinite Curve {27, 12, -23, -32} = (n_elements_width+1) Using Progression p_perp;
        Transfinite Curve {-14, -19} = (n_elements_width+1) Using Progression 1;

      // Top
        Transfinite Curve {-29, -15, 17, 21, 25, 35} = (n_elements_width+1) Using Progression p_perp;

    // Pipe Length
      // Inlet Region
        Transfinite Curve {26, 28, 30} = (n_elements_inlet+1) Using Progression 1;
      // Outlet Region
        Transfinite Curve {33, 31, 34} = (n_elements_outlet+1) Using Progression 1;

    // Semi-circle
      // Perimiter
        Transfinite Curve {1, 7, 5, 3} = (n_elements_perimiter+1) Using Progression 1;
        Transfinite Curve {2, 4, 6, 18, 20} = 2*(n_elements_perimiter+1) Using Progression 1;
        
      // Radius
        Transfinite Curve {16, 13, 8, 9, 10, 11, 22, 24} = (n_elements_radial+1) Using Progression 1;

// Add Transfinite Surfaces
  For i In {1:14}
    Transfinite Surface {i};
    Recombine Surface {i};
  EndFor

// 2D Version
  // Label Boundaries and Surfaces
  // Physical Curve("inlet", 36) = {29, 27};
  //  Physical Curve("outlet", 37) = {35, 32};
  //  Physical Curve("wall", 38) = {33, 34, 24, 20, 16, 30, 26, 1, 2, 3};

  //  Physical Surface("inside", 39) = {6, 12, 11, 5, 2, 3, 1, 7, 8, 9, 10, 4, 13, 14};

  // Create Mesh
  //  Mesh 2;

// 3D Version
   // Extrude to 3D
    Extrude {0, 0, 1} {
      Surface{1:14};
       Layers{n_layers};
       Recombine; 
    }

  // // Label Boundaries and Surfaces
     Physical Surface("inlet", 94) = {54, 50};
     Physical Surface("outlet", 95) = {61, 58};
     Physical Surface("wall", 96) = {62, 57, 52, 55, 24, 27, 20, 46, 42, 35};
     Physical Surface("top_and_bottom", 97) = {23, 2, 5, 33, 37, 6, 12, 56, 11, 53, 19, 1, 40, 7, 43, 8, 9, 47, 10, 49, 4, 29, 3, 26, 13, 60, 14, 63};

     Physical Volume("inside", 93) = {1:14};

  // // Create Mesh
    Mesh 3;
