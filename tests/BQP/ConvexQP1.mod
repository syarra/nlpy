# Ampl model for testing BQP solver
# Simple Convex QP

var x1, <=1, >= -1, := .5;
var x2, <=2, >= -3, := -1/8;

minimize f: x1^2 + 2*x2^2;
