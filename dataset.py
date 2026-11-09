import os
import pickle
import trimesh
import skimage
import numpy as np
from PIL import Image, ImageOps
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from pytorch3d.ops import sample_farthest_points


def mesh_to_point_cloud(mesh, n_points, n_pre_points=8192):
    vertices, faces = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    vertices = vertices[ids]
    return np.asarray(vertices)


def mesh_to_image(mesh, camera_distance=-1.8, front=[1, 1, 1], width=500, height=500, img_size=128):
    import open3d
    vis = open3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    lookat = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    front_array = np.array(front, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    
    eye = lookat + front_array * camera_distance
    right = np.cross(up, front_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(front_array, right)
    rotation_matrix = np.column_stack((right, true_up, front_array)).T
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = -rotation_matrix @ eye

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    image = np.asarray(image)
    image = (image * 255).astype(np.uint8)
    image = skimage.transform.resize(
        image,
        output_shape=(img_size, img_size),
        order=2,
        anti_aliasing=True,
        preserve_range=True).astype(np.uint8)

    return Image.fromarray(image)


class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc, img_size,
                normalize_std_img, noise_scale_img, num_imgs, mode, n_samples=None, ext='stl'):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.n_samples = n_samples
        self.n_points = n_points  
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.normalize_std_img = normalize_std_img
        self.noise_scale_img = noise_scale_img
        self.num_imgs = num_imgs
        self.mode = mode
        if self.split in ['train', 'val']:
            pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
            with open(pkl_path, 'rb') as f:
                self.annotations = pickle.load(f)
        else:
            paths = os.listdir(os.path.join(self.root_dir, self.split))
            self.annotations = [
                {'mesh_path': os.path.join(self.split, f)}
                for f in paths if f.endswith('.stl')
            ]

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        if self.mode == 'pc':
            input_item = self.get_point_cloud(item)
        elif self.mode == 'img':
            input_item = self.get_img(item)
        elif self.mode == 'pc_img':
            if np.random.rand() < 0.5:
                input_item = self.get_point_cloud(item)
            else:
                input_item = self.get_img(item)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        input_item['file_name'] = os.path.basename(item['mesh_path'])[:-4]

        if self.split in ['train', 'val']:
            py_path = item['py_path']
            py_path = os.path.join(self.root_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer

        return input_item

    def get_img(self, item):
        import open3d
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        if self.split in ['train', 'val']:
            mesh.apply_transform(trimesh.transformations.scale_matrix(1 / self.normalize_std_img))
            mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
        else:
            mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
            extent = np.max(mesh.extents)
            if extent > 1e-7:
                    mesh.apply_scale(1.0 / extent)
            mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))


        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.array([255, 255, 136]) / 255.0)
        mesh.compute_vertex_normals()

        fronts = [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
        images = []
        for front in fronts:
            image = mesh_to_image(
                mesh, camera_distance=-0.9, front=front, img_size=self.img_size)
            images.append(image)
                
        images = [ImageOps.expand(image, border=3, fill='black') for image in images]
        if self.num_imgs == 1:
            images = [images[0]]
        elif self.num_imgs == 2:
            images = [Image.fromarray(np.hstack((
                np.array(images[0]), np.array(images[1])
            )))]
        elif self.num_imgs == 4:
            images = [Image.fromarray(np.vstack((
                np.hstack((np.array(images[0]), np.array(images[1]))),
                np.hstack((np.array(images[2]), np.array(images[3])))
            )))]
        else:
            raise ValueError(f'Invalid number of images: {self.num_imgs}')

        input_item = {
            'video': images,
            'description': 'Generate cadquery code'
        }
        return input_item

    def get_point_cloud(self, item):
        mesh = trimesh.load(os.path.join(self.root_dir, item['mesh_path']))
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        
        if self.split in ['train', 'val']:
            point_cloud = point_cloud / self.normalize_std_pc
        else:
            point_cloud = (point_cloud - 0.5) * 2
        
        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh

class CadEditDataset(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc,
                n_samples=None, ext='stl', mesh_root=None):
        super().__init__()
        self.root_dir = root_dir
        self.mesh_root = mesh_root if mesh_root is not None else os.path.join(root_dir, 'mesh')
        self.split = split
        self.n_samples = n_samples
        self.n_points = n_points  
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.annotations = []
        if self.split in ['train', 'test']:
            data_path = os.path.join(self.root_dir, 'final',f'{self.split}.json')
            with open(data_path, "r", encoding="utf-8") as annotation_file:
                self.data_ori = json.load(annotation_file)
            for sample in self.data_ori:
                pc_path = os.path.join(self.mesh_root, sample['original_pic_name'])
                root = Path(pc_path)
                stl_files = sorted(str(p) for p in root.rglob("*.stl"))
                if not stl_files:
                    continue
                pc_path = os.path.join(self.mesh_root, sample['edited_pic_name'])
                root = Path(pc_path)
                stl_files = sorted(str(p) for p in root.rglob("*.stl"))
                if not stl_files:
                    continue
                self.annotations.append(sample)
            print(f"Loaded {len(self.annotations)} samples for split '{self.split}'")

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        sample = self.annotations[index]
        pc_path = os.path.join(self.mesh_root, sample['original_pic_name'])

        pc = self.get_point_cloud(pc_path)

        description = f"Instruction: {sample['instruction']}"
        answer = sample['edited_sequence']
        return {"description": description, "answer": answer, "point_cloud": pc["point_cloud"],"sample":sample,"sequence": sample['original_sequence']}

    def get_point_cloud(self, pc_path):
        root = Path(pc_path)
        item_pc = sorted(str(p) for p in root.rglob("*.stl"))[0]
        mesh = trimesh.load(item_pc)
        # mesh = self._augment_pc(mesh)
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        extent = float(np.max(mesh.extents))
        if extent <= 0:
            raise ValueError(f"Empty source CAD mesh: {item_pc}")
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / extent)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh
    def _normalize_point_cloud(self, point_cloud):
        center = (point_cloud.max(axis=0) + point_cloud.min(axis=0)) / 2.0
        point_cloud = point_cloud - center

        scale = np.max(point_cloud.max(axis=0) - point_cloud.min(axis=0))
        if scale > 1e-8:
            point_cloud = point_cloud / scale

        return point_cloud

class CadEditDataset_query(Dataset):
    def __init__(self, root_dir, split, n_points, normalize_std_pc, noise_scale_pc,
                n_samples=None, ext='stl', mesh_root=None):
        super().__init__()
        self.root_dir = root_dir
        self.mesh_root = mesh_root if mesh_root is not None else os.path.join(root_dir, 'mesh')
        self.split = split
        self.n_samples = n_samples
        self.n_points = n_points  
        self.normalize_std_pc = normalize_std_pc
        self.noise_scale_pc = noise_scale_pc
        self.annotations = []
        if self.split in ['train', 'test']:
            data_path = os.path.join(self.root_dir, 'final',f'{self.split}.json')
            with open(data_path, "r", encoding="utf-8") as annotation_file:
                self.data_ori = json.load(annotation_file)
            for sample in self.data_ori:
                pc_path = os.path.join(self.mesh_root, sample['original_pic_name'])
                root = Path(pc_path)
                stl_files = sorted(str(p) for p in root.rglob("*.stl"))
                if not stl_files:
                    continue
                pc_path = os.path.join(self.mesh_root, sample['edited_pic_name'])
                root = Path(pc_path)
                stl_files = sorted(str(p) for p in root.rglob("*.stl"))
                if not stl_files:
                    continue
                self.annotations.append(sample)
            print(f"Loaded {len(self.annotations)} samples for split '{self.split}'")

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        sample = self.annotations[index]
        pc_path = os.path.join(self.mesh_root, sample['original_pic_name'])

        pc = self.get_point_cloud(pc_path)

        description = f"Instruction: {sample['instruction']}"
        answer = sample['edited_sequence']
        return {"description": description, "answer": answer, "point_cloud": pc["point_cloud"],"sample":sample,"sequence": sample['original_sequence']}

    def get_point_cloud(self, pc_path):
        root = Path(pc_path)
        item_pc = sorted(str(p) for p in root.rglob("*.stl"))[0]
        mesh = trimesh.load(item_pc)
        mesh = self._augment_pc(mesh)
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
        extent = float(np.max(mesh.extents))
        if extent <= 0:
            raise ValueError(f"Empty source CAD mesh: {item_pc}")
        mesh.apply_translation(-center)
        mesh.apply_scale(1.0 / extent)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)

        input_item = {
            'point_cloud': point_cloud,
            'description': 'Generate cadquery code',
        }
        return input_item

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(loc=0, scale=self.noise_scale_pc, size=mesh.vertices.shape)
        return mesh
    def _normalize_point_cloud(self, point_cloud):
        center = (point_cloud.max(axis=0) + point_cloud.min(axis=0)) / 2.0
        point_cloud = point_cloud - center

        scale = np.max(point_cloud.max(axis=0) - point_cloud.min(axis=0))
        if scale > 1e-8:
            point_cloud = point_cloud / scale

        return point_cloud
    
class Text2CADDataset(Dataset):
    def __init__(self, root_dir, split, code_dir='cadquery', n_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.n_samples = n_samples
        self.code_dir = code_dir
        pkl_path = os.path.join(self.root_dir, f'{self.split}.pkl')
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        input_item = {
            'description': item['description'],
            'file_name': item['uid']
        }

        if self.split in ['train', 'val']:
            py_path = f'{item["uid"]}.py'
            py_path = os.path.join(self.root_dir, self.code_dir, py_path)
            with open(py_path, 'r') as f:
                answer = f.read()
            input_item['answer'] = answer
        return input_item
