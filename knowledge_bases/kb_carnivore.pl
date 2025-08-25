nn(net33, [Image], Y, [0,1]) :: t33(Image,Y).
nn(net34, [Image], Y, [0,1]) :: t34(Image,Y).
nn(net37, [Image], Y, [0,1]) :: t37(Image,Y).

inst_has_prop(Image, fish):- t33(Image,1).
inst_has_prop(Image, meat):- t34(Image,1).
inst_has_prop(Image, hunter):- t37(Image,1).
inst_has_prop(Image, not_fish):- t33(Image,0).
inst_has_prop(Image, not_meat):- t34(Image,0).
inst_has_prop(Image, not_hunter):- t37(Image,0).

s1(hunter, meat, predator).
s1(hunter, fish, predator).
s1(not_hunter, meat, not_predator).
s1(not_hunter, fish, not_predator).
s1(not_hunter, not_meat, not_predator).
s1(not_hunter, not_fish, not_predator).

d(Image, Y) :- inst_has_prop(Image, Y).
d(Image, Y) :- s1(Z1, Z2, Y), d(Image, Z1), d(Image, Z2).

is_carnivore(X,0) :- d(X,not_predator).
is_carnivore(X,1) :- d(X,predator).