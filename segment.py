import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    # Load the .obj file
    mesh = o3d.io.read_triangle_mesh(file_path)

    # If it's a mesh, convert it to a point cloud (using vertices)
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        point_cloud = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    else:
        point_cloud = mesh

    return point_cloud

def load_obj_file(file_path):
    # Read the .obj file
    mesh = o3d.io.read_triangle_mesh(file_path)

    # If the .obj has textures, this will apply them
    # if mesh.has_triangle_uvs():
    #     mesh.compute_vertex_normals()
    #     mesh = o3d.geometry.TriangleMesh.create_from_triangle_mesh_within_depth(mesh, depth=1)
    #     mesh.paint_uniform_color([0.9, 0.9, 0.9])  # Optional: set a default color

    return mesh

def segment_sphere(point_cloud, k):
    # Convert to numpy array
    points = np.asarray(point_cloud.points)

    # Heuristic: center of the sphere is the mean of the points
    center = points.mean(axis=0)

    # randomly shift the center but keep it inside the point cloud
    center += np.random.uniform(low=-0.02, high=0.02, size=(3,))
    # Heuristic: start with a small sphere and increase until k% of points are inside
    total_points = len(points)
    target_points = total_points * (k / 100)
    radius = 0.05  # initial radius

    while True:
        # Calculate distances from the center
        distances = np.linalg.norm(points - center, axis=1)

        # Count points inside the sphere
        inside_count = np.sum(distances < radius)

        if inside_count >= target_points:
            break
        else:
            radius += 0.05  # increment radius

    # Create a mask for points outside the sphere
    mask = distances >= radius
    assert mask.sum() > 0, "No points outside the sphere"

    if mask.sum() < target_points:
        mask = distances < radius
        print("Masking inside points instead")

    # Create a new point cloud without the points inside the sphere
    new_points = points[mask]
    new_point_cloud = o3d.geometry.PointCloud()
    new_point_cloud.points = o3d.utility.Vector3dVector(new_points)

    return new_point_cloud, radius, center


# Example usage
# file_path = '/media/exx/T7 Shield/RSS24/VIL2/vil2/assets/google_scanned_objects/bowl/meshes/model.obj'  # Replace with your file path
# file_path = "/media/exx/T7 Shield/RSS24/VIL2/vil2/assets/google_scanned_objects/plate/meshes/model.obj"
# file_path = "/media/exx/T7 Shield/RSS24/VIL2/vil2/assets/google_scanned_objects/spoon/meshes/model.obj"
# file_path = "/media/exx/T7 Shield/RSS24/VIL2/vil2/assets/google_scanned_objects/tea_mug/meshes/model.obj"
file_path = "/media/exx/T7 Shield/RSS24/VIL2/vil2/assets/google_scanned_objects/tea_pot/meshes/model.obj"
k = 10  # Percentage of points to remove
point_cloud = load_point_cloud(file_path)
segmented_cloud, sphere_radius, sphere_center = segment_sphere(point_cloud, k)

# Save the new point cloud if needed
# o3d.io.write_point_cloud('segmented_cloud.pcd', segmented_cloud)
transform = np.eye(4)
transform[:3, 3] = [0.5, 0.5, 0]
segmented_cloud.transform(transform)

print("Original point cloud has %d points." % len(point_cloud.points))
print("Segmented point cloud has %d points." % len(segmented_cloud.points))
o3d.visualization.draw_geometries([point_cloud, segmented_cloud])
# visualize_side_by_side(point_cloud, segmented_cloud)
