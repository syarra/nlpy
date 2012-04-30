# Ampl model for testing BQP solver
# Simple Convex QP

var x1, <=1, >= -1, := -1;
var x2, <=2, >= -2, := -1;
var x3, <=1, >= -1, := -1;

minimize f: x1^2 + 2*x2^2 + x3^2;
