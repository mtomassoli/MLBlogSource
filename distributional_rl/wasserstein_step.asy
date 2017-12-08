
unitsize(2cm);
real width = 8;
real height = 5.5;
real one_y = 5;

draw((0,0)--(width,0), EndArrow);
draw((0,0)--(0,height), EndArrow);

real[] xs = {0.5, 1.2, 2.3, 2.8, 3.2, 3.4, 3.6, 5, 5.4, 5.7, 6, 7};
int N = xs.length;
real step = one_y / N;

path p = (0,0);
for (int i = 0; i < N; ++i)
    p = p -- (xs[i], i*step) -- (xs[i], (i+1)*step);          // adds a step
p = p -- (width, one_y);

path q = (0,0){1,0}..(1,0.5){1,0}..{0.2,0.8}(2.3,1)..(3,2.5){1,0.3}..{0.3,0.5}(5,3.5)..
    (5.5,4){1,1.5}..{1,0}(8,5);
fill(p..reverse(q)..cycle, evenodd+lightcyan);
draw(p, blue+linewidth(1.6));
draw(q, red+linewidth(1.6));

draw((0,one_y)--(8,one_y), dashed);
label("$1$", (0,one_y), W);

for (int i = 0; i < N; ++i) {
    draw((xs[i],0)--(xs[i],(i*step)), dashed);
    label("$y_{" + (string)(i+1) + "}$", xs[i], 2*S, fontsize(10));
}

for (int i = 1; i < N; ++i) {
    draw((0,i*step)--(xs[i-1],i*step), dashed);
    label("$" + (string)i + "q$", (0,i*step), 2*W, fontsize(10));
}