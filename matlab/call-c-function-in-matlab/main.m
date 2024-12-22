% https://www.mathworks.com/help/coder/ug/call-cc-code-from-matlab-code.html

codegen mathOps -args {1, 2} src/add.c;
[added, multed] = mathOps_mex(10, 20);
