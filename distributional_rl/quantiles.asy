
unitsize(1cm);
int N = 10;
real length = 20;

real[] pos;

draw((0,0)--(length,0));

for (int i = 0; i <= 2*N; ++i)
    pos[i] = i*length / (2*N);
    
for (int i = 0; i <= 2*N; i+=2) {
    draw((pos[i],-1)--(pos[i],1));
    string text = "$\frac{" + (string)i + "}{" + string(2*N) + "}$";
    label(text, (pos[i],-1), S, fontsize(18pt));
}

for (int i = 1; i <= N; ++i) {
    real p = pos[2*(i-1) + 1];
    draw((p,-0.5)--(p,0.5));
    string text = "$\hat\tau_{" + (string)i + "}$";
    label(text, (p,-0.5), S, fontsize(18pt));
}

pen fs = fontsize(20pt);
label("$N=" + (string)N + "$", (length*2/8, -3), fs);
label("$\hat\tau_i = \frac{2(i-1)+1}{2N} = \frac{2(i-1)+1}{20}$", (length*5/8, -3), fs);