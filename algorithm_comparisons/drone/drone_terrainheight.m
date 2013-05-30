function [ height ] = drone_terrainheight( model, x )
%drone_terrainheight Return the heigh of the terrain at the current state

map = model.map;

height = 0;
for hh = 1:map.num_hills
    dx = x(1:2) - map.mn(:,hh);
    gaussblob = det(2*pi*map.vr(:,:,hh))^(-1/2)*exp( -0.5*dx'*(map.vr(:,:,hh)\dx) );
    height = height + map.alt(hh)*gaussblob;
end

end

