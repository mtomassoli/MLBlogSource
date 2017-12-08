
unitsize(1cm);
int N = 25;
real length = 20;
real wall_height = 8;

draw((0,0)--(length,0));

real delta = length / (N-1);

real pdf(real x) {
    real[] pis = {1/3, 1/3, 1/3};
    real[] mus = {-3, 0, 5};
    real[] sigmas = {2, 1, 2};
    real sum = 0;
    for (int i = 0; i < 3; ++i)
        sum += pis[i] * exp(-0.5 * ((x-mus[i])/sigmas[i])**2) / (sqrt(2*pi)*sigmas[i]);
    return sum;
}

real radius = 0.2;

for (int i = 0; i < N; ++i) {
    real p = i*delta;
    real x = p - 9;
    draw((p, 0)--(p, 60 * pdf(x)), TrueMargin(0, radius*cm));
    filldraw(circle((p, 60 * pdf(x)), radius), cyan);
    label("$x_{" + (string)(i+1) + "}$", (p, 0), 2.5*S, fontsize(14));
}