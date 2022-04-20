from simgeometry import SimGeometry
import numpy as np

sim = SimGeometry("test_cube.STL")
sim.make_cell_grid(1,0.2)
p0 = np.array([0.,0.,0.])
p1 = np.array([0.,1.,0.])

hit_fi = np.zeros(3)
for fi in sim.mesh.faces:
	triangle = np.array([sim.mesh.vertices[fi[0]], sim.mesh.vertices[fi[1]], sim.mesh.vertices[fi[2]]])
	hit = sim._ray_intersect_triangle(p0,p1,triangle)
	if hit==1:
		hit_fi = fi
		break
#print(hit_fi)
#sim.locate_cut_cells()
sim._clip_segment(p0,p1,sim.mesh.get_face_attribute("face_normal")[95],sim.mesh.faces[95])
sim.show_mesh()
