nn(net10, [Image], Y, [0,1]) :: t10(Image,Y).
nn(net11, [Image], Y, [0,1]) :: t11(Image,Y).
nn(net27, [Image], Y, [0,1]) :: t27(Image,Y).
nn(net43, [Image], Y, [0,1]) :: t43(Image,Y).

inst_has_prop(Image, furry):- t10(Image,1).
inst_has_prop(Image, hairless):- t11(Image,1).
inst_has_prop(Image, swims):- t27(Image,1).
inst_has_prop(Image, water):- t43(Image,1).
inst_has_prop(Image, not_furry):- t10(Image,0).
inst_has_prop(Image, not_hairless):- t11(Image,0).
inst_has_prop(Image, not_swims):- t27(Image,0).
inst_has_prop(Image, not_water):- t43(Image,0).

s1(swims, water, aquatic).
s1(not_swims, not_water, not_aquatic).
s1(aquatic, hairless, hairless_aquatic).
s1(aquatic, furry, furry_aquatic).

d(Image, Y) :- inst_has_prop(Image, Y).
d(Image, Y) :- s1(Z1, Z2, Y), d(Image, Z1), d(Image, Z2).

is_aquatic(X,0) :- d(X,not_aquatic).
is_aquatic(X,1) :- d(X,hairless_aquatic), d(X,not_furry).
is_aquatic(X,2) :- d(X,furry_aquatic), d(X,not_hairless).