
unitsize(2cm);
real width = 8;
real height = 5.5;
real one_y = 5;

draw((0,0)--(width,0), EndArrow);
draw((0,0)--(0,height), EndArrow);

int N = 30;

real step = one_y / N;
real tick = 0.05;
pen small_fs = fontsize(6);

draw((0,one_y)--(8,one_y), dashed);
label("$1$", (0,one_y), W);

path q = (0,0){1,0}..(1,0.5){1,0}..{0.2,0.8}(2.3,1)..(3,2.5){1,0.3}..{0.3,0.5}(5,3.5)..
    (5.5,4){1,1.5}..{1,0}(8,5);

real[] xs;
for (int i = 0; i < N; ++i) {
    real y = one_y * (2*i + 1) / (2*N);
    xs[i] = intersectionpoint((0,y)--(width,y), q).x;
}

path p = (0,0);
for (int i = 0; i < N; ++i)
    p = p -- (xs[i], i*step) -- (xs[i], (i+1)*step);          // adds a step
p = p -- (width, one_y);

fill(p..reverse(q)..cycle, evenodd+lightcyan);
draw(p, blue+linewidth(1.6));
draw(q, red+linewidth(1.6));