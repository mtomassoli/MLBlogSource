
unitsize(1cm);
int N = 10;
real length = 20;
real wall_height = 8;

draw((0,0)--(length,0));

real delta = length / (N-1);

for (int i = 0; i < N; ++i) {
    real p = i*delta;
    draw((p, 0)--(p, wall_height), dashed);
    label("$x_{" + (string)(i+1) + "}$", (p, 0), S, fontsize(16));
}

real prop = 3 / 4;
real height = 6.5;
real radius = 0.2;

int l = floor(N / 2);          // left atom
int r = l + 1;                 // right atom
pair l_pos = (l*delta, (1-prop)*height);
pair r_pos = (r*delta, prop*height);

pair atom_pos = (l_pos.x + prop*delta, height);

filldraw(circle(atom_pos, radius), lightgreen);
filldraw(circle(l_pos, radius), lightgrey);
filldraw(circle(r_pos, radius), lightgrey);

margin margin = TrueMargin(radius*cm, radius*cm);
draw(atom_pos -- l_pos, Arrow, margin);
draw(atom_pos -- r_pos, Arrow, margin);

real y = atom_pos.y + 0.5;
draw((l_pos.x, y) -- (atom_pos.x, y), dashed + blue, Arrows, Bars);
draw((atom_pos.x, y) -- (r_pos.x, y), dashed + red, Arrows, Bars);

real x = l_pos.x - 0.5;
draw((x, 0) -- (x, l_pos.y), dashed + red, Arrows, Bars);
real x = r_pos.x + 0.5;
draw((x, 0) -- (x, r_pos.y), dashed + blue, Arrows, Bars);