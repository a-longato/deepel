nn(net17, [I], Y, [0,1]) :: t17(I,Y).
nn(net18, [I], Y, [0,1]) :: t18(I,Y).
nn(net27, [I], Y, [0,1]) :: t27(I,Y).
nn(net32, [I], Y, [0,1]) :: t32(I,Y).
nn(net35, [I], Y, [0,1]) :: t35(I,Y).
nn(net42, [I], Y, [0,1]) :: t42(I,Y).
nn(net43, [I], Y, [0,1]) :: t43(I,Y).

inst_has_prop(I, c_hands) :- t17(I,1).
inst_has_prop(I, c_hooves) :- t18(I,1).
inst_has_prop(I, c_swims) :- t27(I,1).
inst_has_prop(I, c_quadrapedal):- t32(I,1).
inst_has_prop(I, c_vegetation) :- t35(I,1).
inst_has_prop(I, c_ground) :- t42(I,1).
inst_has_prop(I, c_water) :- t43(I,1).

inst_has_prop(I, n_hands) :- t17(I,0).
inst_has_prop(I, n_hooves) :- t18(I,0).
inst_has_prop(I, n_swims) :- t27(I,0).
inst_has_prop(I, n_water) :- t43(I,0).


s1(c_ground, c_quadrapedal, c_terrestrial).
s2(c_vegetation, r_sustained_by, c_plants).
s3(r_sustained_by, c_plants, c_herbivore).
s1(c_terrestrial, c_herbivore, c_grazer).
s2(c_hooves, r_has_part, c_hoof_part).
s3(r_has_part, c_hoof_part, c_ungulate_form).

s1(c_terrestrial, c_ungulate_form, category_ungulate).

s0(n_hooves, n_ungulate_form).


s1(c_swims, c_water, c_water_mobile).
s2(c_water_mobile, r_moves_via, c_swim_mode).
s3(r_moves_via, c_swim_mode, c_aquatic_nature).

s1(c_aquatic_nature, n_ungulate_form, category_aquatic).

s1(n_swims, n_water, n_water_mobile).
s2(n_water_mobile, r_moves_via, n_swim_mode).
s3(r_moves_via, n_swim_mode, n_aquatic_nature).


s2(c_hands, r_has_part, c_hand_part).
s3(r_has_part, c_hand_part, c_primate_form_1).
s1(n_ungulate_form, n_aquatic_nature, c_primate_form_2).

s1(c_primate_form_1, c_primate_form_2, category_primate).


d(I, C) :- inst_has_prop(I, C).
d(I, C) :- s0(A, C), d(I, A).
d(I, C) :- s1(A, B, C), d(I, A), d(I, B).

d_exists(I, R, B) :- s2(A, R, B), d(I, A).
d_exists(I, R, C) :- d_exists(I, R, B), s0(B, C).
d(I, A) :- s3(R, B, A), d_exists(I, R, B).

predict(I,0) :- d(I, category_primate).
predict(I,1) :- d(I, category_aquatic).
predict(I,2) :- d(I, category_ungulate).
predict(I,3) :- d(I, n_hands), d(I, n_ungulate_form), d(I, n_aquatic_nature).