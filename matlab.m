function oscillo(Fe, so, nb, nfig)
    s = zeros(nb);
    s(1:length(so)) = so;
    t = 0:1/Fe:(length(s)-1)/Fe;
    fx = (-nb/2:1:nb/2-1)*(Fe/nb);
    X = fft(s, nb);
    Xn = abs(X/nb);
    Y = fftshift(Xn);
    figure(nfig);
    subplot(121);
    plot(t, s, 'ro-');
    subplot(122);
    plot(fx, Y, 'bo-');
end
