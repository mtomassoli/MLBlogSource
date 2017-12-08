
unitsize(2cm);
real width = 8;
real height = 5.5;

draw((0,0)--(width,0), EndArrow);
draw((0,0)--(0,height), EndArrow);

path p = (0,0){1,0}..(1,0.3){1,0}..{0.2,0.7}(2.0,1)..(2.5,1.5){1,0.3}..{0.3,1}(3.5,2)..{0.3,0.8}(3.8,3.1)..
    (4.4,3.5){1,1}..(4.8,3.7)..(6,4){1,1.5}..{1,0}(8,5);
path q = (0,0){1,0}..(1,0.5){1,0}..{0.2,0.8}(2.3,1)..(3,2.5){1,0.3}..{0.3,0.5}(5,3.5)..
    (5.5,4){1,1.5}..{1,0}(8,5);
fill(p..reverse(q)..cycle, evenodd+lightcyan);
draw(p, blue+linewidth(1.6));
draw(q, red+linewidth(1.6));

real one_y = 5;
draw((0,one_y)--(8,one_y), dashed);
label("$1$", (0,one_y), W, fontsize(15));

real y = 2;
path seg = (0,y)--(8,y);
pair l_int = intersectionpoint(seg, q);
pair r_int = intersectionpoint(seg, p);
draw(l_int--r_int, linewidth(3), TrueMargin(1,1));

draw((0,l_int.y)--l_int, dashed);
label("$\tau$", (0,l_int.y), W, fontsize(15));

draw((l_int.x,y)--(l_int.x,0), dashed);
label("$F_X^{-1}(\tau)$", (l_int.x,0), 2*S, fontsize(15));

draw((r_int.x,y)--(r_int.x,0), dashed);
label("$F_Y^{-1}(\tau)$", (r_int.x,0), 2*S, fontsize(15));