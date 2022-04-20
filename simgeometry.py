import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import pymesh
import math
import pyvista as pv


class SimGeometry:

	def __init__(self, geo_file, inflow_triangles=None, outflow_triangles=None):
		'''
		@ geo_file: location of geometry file containing triangle mesh 3D model. .obj, .ply, .off, .stl, .mesh, .node, .poly and .msh supported

		'''
		self.mesh = pymesh.load_mesh(geo_file)
		self.mesh.add_attribute("face_normal")
		self.mesh_2 = pv.make_tri_mesh(self.mesh.vertices, self.mesh.faces)
		print(self.mesh.get_face_attribute("face_normal"))
		if inflow_triangles is not None:
			self.inflow_triangles = inflow_triangles
		if outflow_triangles is not None:
			self.outflow_triangles = outflow_triangles

		self.cell_len = None
		self.cell_grid = None
		self.cut_cell_intersecting_faces = None
		self.cell_volumes = None
		self.active_cell_bool_array = None
		self.cut_cell_bool_array = None
	
	
	def make_cell_grid(self, lamb, cell_frac):
		xmin, xmax = np.min(self.mesh.vertices[:,0]), np.max(self.mesh.vertices[:,0])
		ymin, ymax = np.min(self.mesh.vertices[:,1]), np.max(self.mesh.vertices[:,1])
		zmin, zmax = np.min(self.mesh.vertices[:,2]), np.max(self.mesh.vertices[:,2])
		lamb = 1 #mean free path
		cell_frac = 0.10
		self.cell_len = lamb*cell_frac

		x = np.arange(xmin,xmax+self.cell_len, self.cell_len)
		y = np.arange(ymin,ymax+self.cell_len, self.cell_len)
		z = np.arange(zmin,zmax+self.cell_len, self.cell_len)
		X,Y,Z = np.meshgrid(x,y,z)

		self.cell_grid = np.array([X,Y,Z])
		self.cell_volumes = np.ones((len(x)-1,len(y)-1,len(z)-1))*self.cell_len**3
		self.active_cell_bool_array = np.ones((len(x)-1,len(y)-1,len(z)-1))
		self.cut_cell_bool_array = np.zeros((len(x)-1,len(y)-1,len(z)-1))

	def _signed_tetrahedron_volume(self,m,n,r,s):
		'''
		returns the signed volume of the tetrahedron formed by the four points m,n,r,s
		'''
		rec_matrix = np.array([m,n,r,s])
		matrix = np.hstack((rec_matrix, np.ones((4,1))))
		return np.linalg.det(matrix)/6

	def _ray_intersect_triangle(self,p0,p1,triangle):
		Vp0 = self._signed_tetrahedron_volume(p0,triangle[0],triangle[1],triangle[2])
		Vp1 = self._signed_tetrahedron_volume(p1,triangle[0],triangle[1],triangle[2])
		if np.sign(Vp0)==np.sign(Vp1):
			if (Vp0==Vp1 and Vp0==0.0):
				return 1
			else:
				return 0
		Vp0_ab_p1 = self._signed_tetrahedron_volume(p0,triangle[0],triangle[1],p1)
		Vp0_bc_p1 = self._signed_tetrahedron_volume(p0,triangle[1],triangle[2],p1)
		Vp0_ca_p1 = self._signed_tetrahedron_volume(p0,triangle[2],triangle[0],p1)
		if np.sign(Vp0_ab_p1)==np.sign(Vp0_bc_p1) and np.sign(Vp0_ab_p1)==np.sign(Vp0_ca_p1):
			return 1
		else:
			return 0

	def _triangle_intersect_cube(self, triangle, cube):

		# check if any of the triangle vertices are inside the cube
		for p in triangle:
			greater = p > cube
			inside =  ~np.any(np.equal(*greater))
			if inside:
				#print("SMERGEN")
				return 1
		
		# check if all the verticies are on the same side as one of the faces of the cube
		# (and oppositve the side of the rest of the cube)
		# loop through three surfaces connected to lower corner and upper corner of the cube each
		for s in range(3):
			group1_vertices = np.ones((3,3))*cube[0]
			group2_vertices = np.ones((3,3))*cube[1]
			group1_vertices[1,s], group1_vertices[2,(s+1)%3] = (1,1)
			group2_vertices[1,s], group1_vertices[2,(s+1)%3] = (-1,-1)
			if np.all(triangle[:,(s+2)%3] < group1_vertices[:,(s+2)%3]):
				#print("dERGEN")
				return 0
			if np.all(triangle[:,(s+2)%3] > group2_vertices[:,(s+2)%3]):
				#print("HARGEN")
				return 0

		# checks are inconclusive to this point, so we procede with big guns
		# loop through all the edges. This can be done using the 8 vertices looped through earlier
		intersections = []
		for s in range(3):
			group1_vertices = np.ones((3,3))*cube[0]
			group2_vertices = np.ones((3,3))*cube[1]
			intersections.append(self._ray_intersect_triangle(group1_vertices[0],group1_vertices[s],triangle))
			intersections.append(self._ray_intersect_triangle(group1_vertices[s], group2_vertices[(s+1)%3], triangle))
			intersections.append(self._ray_intersect_triangle(group2_vertices[0],group2_vertices[s],triangle))
			intersections.append(self._ray_intersect_triangle(group2_vertices[s], group1_vertices[(s+1)%3], triangle))
			for p0 in triangle:
				for p1 in triangle:
					if np.all(p0==p1): continue
					intersections.append(self._ray_intersect_triangle(p0,p1,group1_vertices))
					intersections.append(self._ray_intersect_triangle(p0,p1,group2_vertices))
		return np.any(intersections)
				    

	#def get_all_traingles_intersecting_cell(cell):

	def locate_cut_cells(self):
		intersecting_faces_matrix = []
		for i in range(len(self.cut_cell_bool_array)):
			intersecting_faces_matrix.append([])
			for j in range(len(self.cut_cell_bool_array[i])):
				intersecting_faces_matrix[i].append([])
				for k in range(len(self.cut_cell_bool_array[i,j])):
					intersecting_faces_matrix[i][j].append([])
					intersecting_faces = []
					for f in self.mesh.faces:
						triangle = np.array([self.mesh.vertices[f[i]] for i in range(len(f))])
						cube = np.array([self.cell_grid[:,i,j,k], self.cell_grid[:,i,j,k]+self.cell_len] )
						if self._triangle_intersect_cube(triangle,cube):
							self.cut_cell_bool_array[i,j,k] = 1
							intersecting_faces.append(f)
					intersecting_faces_matrix[i][j][k] = intersecting_faces
		self.cut_cell_intersecting_faces = np.array(intersecting_faces_matrix, dtype='object')

	
	def _segment_triangle_pierce_point(self,p0,p1,normal,p_plane):
		"""
		clip line segment at plan defined by plane normal and one point in plane (p_plane)
		"""
		c = np.dot(normal,p_plane)
		d0, d1 = (np.dot(normal,p0)-c, np.dot(normal,p1)-c)
		q = p0+d0/(d0-d1)*(p1-p0)
		return q

	def clip_triangle_in_cube(self,triangle,cube):
		bounds = np.array([cube[0,0], cube[1,0], cube[0,1], cube[1,1], cube[0,2], cube[1,2],])
		clipped = 


	#def add_outgas_surface(self, triangleid):

	#def add_pumpout_surface(self, triangleid, sticking_coefficient=1.0):

	def _set_axes_equal(self, ax: plt.Axes):
	    """Set 3D plot axes to equal scale.

	    Make axes of 3D plot have equal scale so that spheres appear as
	    spheres and cubes as cubes.  Required since `ax.axis('equal')`
	    and `ax.set_aspect('equal')` don't work on 3D.
	    """
	    limits = np.array([
	        ax.get_xlim3d(),
	        ax.get_ylim3d(),
	        ax.get_zlim3d(),
	    ])
	    origin = np.mean(limits, axis=1)
	    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
	    self._set_axes_radius(ax, origin, radius)

	def _set_axes_radius(self, ax, origin, radius):
	    x, y, z = origin
	    ax.set_xlim3d([x - radius, x + radius])
	    ax.set_ylim3d([y - radius, y + radius])
	    ax.set_zlim3d([z - radius, z + radius])

	def show_mesh(self):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((1,1,1))
		ax.plot_trisurf(self.mesh.vertices[:,0], self.mesh.vertices[:,1], self.mesh.vertices[:,2], triangles=self.mesh.faces,edgecolor=[[0,0,0]], linewidth=0.5,alpha=0.0,shade=False)
		ax.voxels(self.cell_grid[0], self.cell_grid[1], self.cell_grid[2], self.cut_cell_bool_array, edgecolor=[[1,0,0]], linewidth=0.25, alpha=0.1)
		ax.set_axis_off()
		self._set_axes_equal(ax)
		plt.show()

