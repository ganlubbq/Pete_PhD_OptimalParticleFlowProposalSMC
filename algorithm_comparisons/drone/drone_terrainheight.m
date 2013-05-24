function [ height ] = drone_terrainheight( model, x )
%drone_terrainheight Return the heigh of the terrain at the current state

map = model.map;

height = 0;
for hh = 1:map.num_hills
    height = height + map.alt(hh)*mvnpdf(x(1:2)', map.mn(:,hh)', map.vr(:,:,hh));
end

end

