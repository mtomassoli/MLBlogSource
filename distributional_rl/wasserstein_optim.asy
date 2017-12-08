
unitsize(2cm);
real width = 8;
real height = 5.5;
real one_y = 5;

draw((0,0)--(width,0), EndArrow);
draw((0,0)--(0,height), EndArrow);

int N = 12;

real step = one_y / N;
real tick = 0.05;
pen small_fs = fontsize(6);

for (int i = 1; i < N; ++i) {
    real y = i*step;
    draw((-tick, y)--(tick, y));
    label("$" + (string)i + "q$", (0,y), 2*W, small_fs);
}
draw((-tick,one_y)--(tick,one_y));
label("$1$", (0,one_y), 2*W, small_fs);

path q = (0,0){1,0}..(1,0.5){1,0}..{0.2,0.8}(2.3,1)..(3,2.5){1,0.3}..{0.3,0.5}(5,3.5)..
    (5.5,4){1,1.5}..{1,0}(8,5);

real[] xs;
for (int i = 0; i < N; ++i) {
    real y = one_y * (2*i + 1) / (2*N);
    xs[i] = intersectionpoint((0,y)--(width,y), q).x;
    draw((0,y)--(xs[i],y), dashed);
    label("$\hat\tau_{" + string(i+1) + "}$", (0,y), 2*W);
}

path p = (0,0);
for (int i = 0; i < N; ++i)
    p = p -- (xs[i], i*step) -- (xs[i], (i+1)*step);          // adds a step
p = p -- (width, one_y);

fill(p..reverse(q)..cycle, evenodd+lightcyan);
draw(p, blue+linewidth(1.6));
draw(q, red+linewidth(1.6));