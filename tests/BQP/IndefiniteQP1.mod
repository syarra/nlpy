# Ampl model for testing BQP solver
# Simple Convex QP

#var x1, <=1/2, >= -1, := -1/4;
var x1, <=1/2, >= -1, := 1/4;
var x2, <=1, >= -2, := 1/8;

minimize f: -x1^2 + 3*x2^2;
