nn(net18, [Image], Y, [0,1]) :: t18(Image,Y).
nn(net22, [Image], Y, [0,1]) :: t22(Image,Y).
nn(net32, [Image], Y, [0,1]) :: t32(Image,Y).
nn(net35, [Image], Y, [0,1]) :: t35(Image,Y).
nn(net42, [Image], Y, [0,1]) :: t42(Image,Y).

inst_has_prop(Image, hooves):- t18(Image,1).
inst_has_prop(Image, horns):- t22(Image,1).
inst_has_prop(Image, quadrapedal):- t32(Image,1).
inst_has_prop(Image, vegetation):- t35(Image,1).
inst_has_prop(Image, ground):- t42(Image,1).
inst_has_prop(Image, not_hooves):- t18(Image,0).
inst_has_prop(Image, not_horns):- t22(Image,0).
inst_has_prop(Image, not_quadrapedal):- t32(Image,0).
inst_has_prop(Image, not_vegetation):- t35(Image,0).
inst_has_prop(Image, not_ground):- t42(Image,0).

s1(ground, quadrapedal, terrestrial).
s0(vegetation, herbivore).

s1(terrestrial, herbivore, grazer).
s2(grazer, hunted_by, predator).
s3(hunted_by, predator, prey).

s1(prey, hooves, ungulate).
s1(ungulate, horns, horned_ungulate).
s1(ungulate, not_horns, not_horned_ungulate).

d(Image, Y) :- inst_has_prop(Image, Y).
d(Image, Y) :- d(Image, Z), s0(Z, Y).
d(Image, Y) :- s1(Z1, Z2, Y), d(Image, Z1), d(Image, Z2).

d_exists(Image, R, B) :- d(Image, A), s2(A, R, B).
d_exists(Image, R, C) :- d_exists(Image, R, B), s0(B, C).
d(Image, B) :- d_exists(Image, R, C), s3(R, C, B).

is_ungulate(X,0) :- d(X,not_hooves).
is_ungulate(X,1) :- d(X,not_horned_ungulate).
is_ungulate(X,2) :- d(X,horned_ungulate).
