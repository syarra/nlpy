# Ampl model for testing BQP solver
# Simple Convex QP

var x1, <=2, >= 1, := 3/2;
var x2, <=3, >= 1, := 3;

minimize f: x1^2 + 2*x2^2;
