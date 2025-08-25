nn(net32, [Image], Y, [0,1]) :: t32(Image,Y).
nn(net42, [Image], Y, [0,1]) :: t42(Image,Y).

inst_has_prop(Image, quadrupedal):- t32(Image,1).
inst_has_prop(Image, ground):- t42(Image,1).
inst_has_prop(Image, not_ground):- t42(Image,0).

s1(ground, quadrupedal, terrestrial).
s0(not_ground, not_terrestrial).

d(Image, Y) :- inst_has_prop(Image, Y).
d(Image, Y) :- d(Image, Z), s0(Z, Y).
d(Image, Y) :- s1(Z1, Z2, Y), d(Image, Z1), d(Image, Z2).

is_terrestrial(X,0) :- d(X,not_terrestrial).
is_terrestrial(X,1) :- d(X,terrestrial).