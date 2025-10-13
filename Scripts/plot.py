
load 3u5t_A.pdb, ref
load 3u5t_A_pre.pdb, pred

remove solvent
remove resn HOH
remove not polymer.protein

align pred, ref
rms_cur pred, ref

color green, ref
spectrum b, blue_white_red, pred, minimum=0, maximum=3

show cartoon, all
hide lines, all
bg_color white
set cartoon_transparency, 0.3, ref

orient
ray 1600, 1600
png 3u5t_A_compare.png
