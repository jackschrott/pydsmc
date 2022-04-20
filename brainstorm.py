import numpy as np
import matplotlib.pyplot as plt
import pymesh
from mpl_toolkits.mplot3d import Axes3D

def ray_intersect_triangle(p0, p1, triangle):
    # Tests if a ray starting at point p0, in the direction
    # p1 - p0, will intersect with the triangle.
    #
    # arguments:
    # p0, p1: numpy.ndarray, both with shape (3,) for x, y, z.
    # triangle: numpy.ndarray, shaped (3,3), with each row
    #     representing a vertex and three columns for x, y, z.
    #
    # returns: 
    #    0.0 if ray does not intersect triangle, 
    #    1.0 if it will intersect the triangle,
    #    2.0 if starting point lies in the triangle.
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)
    
    # Here is the main difference with the code in the link.
    # Instead of returning if the ray is in the plane of the 
    # triangle, we set rI, the parameter at which the ray 
    # intersects the plane of the triangle, to zero so that 
    # we can later check if the starting point of the ray
    # lies on the triangle. This is important for checking 
    # if a point is inside a polygon or not.
    
    if (b == 0.0):
        # ray is parallel to the plane
        if a != 0.0:
            # ray is outside but parallel to the plane
            return 0
        else:
            # ray is parallel and lies in the plane
            rI = 0.0
    else:
        rI = a / b
    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom
    
    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom
    
    if (ti < 0.0) | (si + ti > 1.0):
        return 0
    if (rI == 0.0):
        # point 0 lies ON the triangle. If checking for 
        # point inside polygon, return 2 so that the loop 
        # over triangles can stop, because it is on the 
        # polygon, thus inside.
        return 2
    return 1

def set_axes_equal(ax: plt.Axes):
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
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def get_triangle(mesh, i):
	fi = mesh.faces[i]
	triangle = np.array([mesh.vertices[fi[0]], mesh.vertices[fi[1]], mesh.vertices[fi[2]]])
	return triangle

mesh = pymesh.load_mesh("test_cube.STL")

p0 = np.array([0.0,0.0,0.0])
p1 = np.array([1.,0.0,0.0])
p = np.array([p0,p1])



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((1,1,1))

hit_fi = np.zeros(3)
for fi in mesh.faces:
	triangle = np.array([mesh.vertices[fi[0]], mesh.vertices[fi[1]], mesh.vertices[fi[2]]])
	hit = ray_intersect_triangle(p0,p1,triangle)
	if hit==1:
		hit_fi = fi
		break

print(hit_fi)
hit_vi = np.zeros((3,3))

for i in range(len(hit_vi)):
	for j in range(len(hit_vi[i])):
		#print(mesh.vertices[hit_fi[i]][j])
		hit_vi[i,j] = mesh.vertices[hit_fi[i]][j]

print(hit_vi)
ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces,edgecolor=[[0,0,0]], linewidth=0.5,alpha=0.0,shade=False)
ax.plot_trisurf(hit_vi[:,0], hit_vi[:,1], hit_vi[:,2], triangles=[[0,1,2]],edgecolor=[[1,0,0]], linewidth=1.0, alpha=0.5)
ax.plot(p[:,0],p[:,1],p[:,2],'-r')
ax.set_axis_off()
set_axes_equal(ax)
plt.show()