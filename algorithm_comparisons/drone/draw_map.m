function draw_map(map)
% Draw the terrain map for the drone model

x_rng = -1000:10:1000;
Nx = length(x_rng);

height = zeros(Nx^2,1);

x1coord = repmat(x_rng,Nx,1);
x1coord = x1coord(:);
x2coord = repmat(x_rng,Nx,1)';
x2coord = x2coord(:);
coord = [x1coord x2coord];

for hh = 1:map.num_hills
    height = height + map.alt(hh)*mvnpdf(coord, map.mn(:,hh)', map.vr(:,:,hh));
end

figure, contour(x_rng, x_rng, reshape(height,Nx,Nx) );
figure, surf(x_rng, x_rng, reshape(height,Nx,Nx) );
shading interp;

end