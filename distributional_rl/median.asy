
unitsize(1cm);
real length = 20;
real height = 5;

draw((0,0)--(length,0));

real radius = 0.2;

real red_x = length * 6/8;
filldraw(circle((red_x,0), radius), red);
label("$\theta$", (red_x,0), 2*S, fontsize(18));
draw((red_x,0) -- (red_x,height), dashed, TrueMargin(radius*cm, 0));

real[] pos_xs = {0.1, 0.2, 0.25, 0.32, 0.35, 0.37, 0.4, 0.48, 0.5, 0.6, 0.65, 0.7,
                 0.8, 0.83, 0.93, 0.95};

real radius = 0.35;

for (int i = 0; i < pos_xs.length; ++i) {
    real x = pos_xs[i] * length;
    string text;
    pen color;
    if (x < red_x) {
        text = "$-\alpha$";
        color = lightgreen;
    }
    else
    {
        text = "$+\alpha$";
        color = lightcyan;
    }
    filldraw(circle((x,0), radius), color);
    label(text, (x,0), fontsize(11));
}

real samples_prop = 3/4;
real arrow_units = 0.5;

real x = red_x / 2;
real dx = samples_prop / 2 * length * arrow_units;
draw((x-dx, height/2) -- (x+dx, height/2), BeginArrow);
label("$-12\alpha$", (x, height/2 + 0.5), fontsize(14));

x = (red_x + length) / 2;
real dx = (1 - samples_prop) / 2 * length * arrow_units;
draw((x-dx, height/2) -- (x+dx, height/2), EndArrow);
label("$+4\alpha$", (x, height/2 + 0.5), fontsize(14));