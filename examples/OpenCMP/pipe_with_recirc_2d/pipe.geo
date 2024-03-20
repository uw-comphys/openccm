SetFactory("OpenCASCADE");
// force .msh version 2
Mesh.MshFileVersion=2.2;

// Variables
  // Space between the outer curve and the inset curve
  inset = 0.005;

  // Radius of the outer circle
  r   = 1.0;
  // Radius of the inset circle
  r_2 = r - inset; 

  // Inlet length
  len_in  = 1.0;
  // Outlet length
  len_out = 5.0;
  // Total length of the pipe
  len_total = len_in + 2*r + len_out;

  // Height of the pipe
  height_pipe = 0.5;

  // Fake offset, used to see the creation of the two inset pieces
  // it's automatically removed by using a Translate
  offset_fake = 2;

  // Number of elements along the inlet and outlets
  n_elements_vert_int_out = 10;
  // Number of elements along the top length of pipe
  n_elements_top    = 40;
  // Number of elements along the bottom inlet length of pipe
  n_elements_inlet  = 15;
  // Number of elements along the bottom outlet length of pipe
  n_elements_outlet = 35;
  // Number of elements along the half of the circle
  n_elements_circle = 35;
  // Number of elements in boundary layer
  n_elements_boundary =  2;
  // Number of elements along the line for capturing the vortex
  n_elements_vortex_line = 30;


// Create outer curve
  // Create the half-circle
    // Create a full circle and add a surface to it
    Circle(1) = {
      len_in + r, 0, 0, // Center of circle
      r,                // Radius
      0,                // Start angle
      2*Pi              // End angle
    };
    Curve Loop(1) = {1};
    Plane Surface(1) = {1};

    // Create the rectangle used to cut the circle
    // The cut circle needs to overlap with the final rectangle, BUT only go outside of it on the one side which we want to keep.
    Rectangle(2) = {
      len_in, inset,  0, // Bottom left corner (x, y, z)
      2*r,    r,      0  // Size (dx, dy, dz)
    };
    
    // Cut the circle with the cutting rectangle
    BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
  
  // Create the rectangle
  Rectangle(3) = {
    0,          0,            0,  // Bottom left corner (x, y, z)
    len_total,  height_pipe,  0   // Size (dx, dy, dz)
  };

  // Combine the half-circle and the rectangle
  BooleanUnion{ Surface{1}; Delete; }{ Surface{3}; Delete; }


// Add extra refinement line
  // // End point
  // Point(19) = {2.5, -0.35, 0, 1.0};

  // // Line
  // Line(25) = {8, 19};

  // // Add to surface
  // Line{25} In Surface{1};

// Create the transfinite curves & planes
  // The circle arc
  Transfinite Curve { 9} = n_elements_circle+1 Using Progression 0.97;
  // The top lines
  Transfinite Curve { 6} = n_elements_top+1 Using Progression 1;
  // The bottom left lines
  Transfinite Curve { 8} = n_elements_inlet+1 Using Progression 0.9;
  // The bottom right lines
  Transfinite Curve {10} = n_elements_outlet+1 Using Progression 1.1;
  // The inlet
  Transfinite Curve { 7} = n_elements_vert_int_out+1 Using Progression 1;
  // The outlet
  Transfinite Curve { 5} = n_elements_vert_int_out+1 Using Progression 1;
  // Extra Refinement Line
  Transfinite Curve {25} = n_elements_vortex_line+1 Using Progression 1;

// Add physical surfaces
  Physical Curve("wall") = {6, 8, 9, 10};
  Physical Curve("outlet") = {5};
  Physical Curve("inlet") = {7};

  Physical Surface("inside") = {1};

Mesh 2;
RefineMesh;
// RefineMesh;
// RefineMesh;
// RefineMesh;
Mesh 2;
