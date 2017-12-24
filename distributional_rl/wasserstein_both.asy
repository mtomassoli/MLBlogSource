
unitsize(2cm);

real width = 9.5;
real height = 5.5;

int N = 60;

real one_y = 5;
real sample_radius = 0.05;

path get_step_func(real[] xs) {
    draw((0,0)--(width,0), EndArrow);
    draw((0,0)--(0,height), EndArrow);

    int N = xs.length;
    real step = one_y / N;

    path p = (0,0);
    for (int i = 0; i < N; ++i)
        p = p -- (xs[i], i*step) -- (xs[i], (i+1)*step);          // adds a step
    p = p -- (width, one_y);

    return p;
}

void draw_samples(real[] xs, pen my_pen=black) {
    for (int i = 0; i < xs.length; ++i)
        fill(circle((xs[i],-0.25),sample_radius), my_pen);
}

real[] xs = {0.5, 1.2, 2.3, 2.8, 3.2, 3.4, 3.6, 5, 5.4, 5.7, 6, 7};
path p = get_step_func(xs);
real[] ys = {0.9, 1.9, 2.6, 2.7, 3.9, 3.95, 4, 4.3, 4.7, 6, 6.5, 6.9};
path q = get_step_func(ys);

fill(p..reverse(q)..cycle, evenodd+lightcyan);
draw(p, blue+linewidth(1.6));
draw(q, red+linewidth(1.6));

draw((0,one_y)--(8,one_y), dashed);
label("$1$", (0,one_y), W);

real radius = 0.1;
pen fs = fontsize(16);

real y = height * 3/4;
filldraw(circle((1,y), radius), blue);
label("$Z(x,a)$", (1,y-radius/4), 5*E, fs);

y -= height * 1/10;
filldraw(circle((1,y), radius), red);
label("$r+\gamma Z(x',a^*)$", (1,y-radius/4), 5*E, fs);