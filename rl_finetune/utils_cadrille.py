import os
import time

os.environ["PYGLET_HEADLESS"] = "True"
from pathlib import Path
from multiprocessing import TimeoutError, get_context
import numpy as np
import trimesh
import sys
import tempfile
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


REFERENCE_FIELD_CACHE = {}
REFERENCE_FIELD_CACHE_LIMIT = 128


def _sample_mesh_surface(mesh, n_points):
    if n_points <= 0 or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Cannot sample an empty or non-finite mesh")
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(points, dtype=np.float32)


def reference_transform(gt_mesh, ori_mesh):
    bounds = np.concatenate([gt_mesh.bounds, ori_mesh.bounds], axis=0)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    extent = float(np.max(upper - lower))
    if not np.isfinite(bounds).all() or extent <= 0:
        raise ValueError("Invalid reference mesh bounds")
    return (lower + upper) / 2.0, 1.0 / extent


def apply_reference_transform(mesh, center, scale, native_scale=1.0):
    mesh.apply_scale(native_scale)
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh


def normalize_mesh_triplet(pred_mesh, gt_mesh, ori_mesh, prediction_scale=1.0):
    # Scale exported predictions back to dataset units before the shared transform.
    center, scale = reference_transform(gt_mesh, ori_mesh)
    apply_reference_transform(pred_mesh, center, scale, prediction_scale)
    apply_reference_transform(gt_mesh, center, scale)
    apply_reference_transform(ori_mesh, center, scale)
    return pred_mesh, gt_mesh, ori_mesh


def _make_voxel_grid(resolution, margin=0.0):
    if resolution < 2:
        raise ValueError("Voxel resolution must be at least two")
    axis = np.linspace(margin, 1.0 - margin, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return grid, float(axis[1] - axis[0]), float(axis[0])


def _mesh_occupancy_tsdf(mesh, grid, tau_phi, chunk_size=8192):
    if tau_phi <= 0:
        raise ValueError("TSDF truncation distance must be positive")
    points = grid.reshape(-1, 3)
    occupancy = np.empty(len(points), dtype=np.float32)
    tsdf = np.empty(len(points), dtype=np.float32)
    try:
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            occupancy[start:start + chunk_size] = mesh.contains(chunk)
            distance = trimesh.proximity.signed_distance(mesh, chunk)
            tsdf[start:start + chunk_size] = np.clip(distance, -tau_phi, tau_phi)
    except Exception as exc:
        raise RuntimeError(
            f"LFM occupancy/TSDF failed: watertight={mesh.is_watertight}, "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not np.isfinite(tsdf).all():
        raise ValueError("Non-finite TSDF values")
    return occupancy.reshape(grid.shape[:-1]), tsdf.reshape(grid.shape[:-1])


def _build_reference_field(gt_mesh, ori_mesh, n_points=2000, tau_phi=0.01,
                           grid_resolution=64, cache_key=None):
    if cache_key is not None and cache_key in REFERENCE_FIELD_CACHE:
        return REFERENCE_FIELD_CACHE[cache_key]
    grid, spacing, origin = _make_voxel_grid(grid_resolution)
    ori_occ, ori_tsdf = _mesh_occupancy_tsdf(ori_mesh, grid, tau_phi)
    target_occ, target_tsdf = _mesh_occupancy_tsdf(gt_mesh, grid, tau_phi)
    delta_occ = np.abs(target_occ - ori_occ)
    delta_tsdf = np.minimum(np.abs(target_tsdf - ori_tsdf) / tau_phi, 1.0)
    w_edit = (1.0 - (1.0 - delta_occ) * (1.0 - delta_tsdf)).astype(np.float32)
    context = {
        "gt_points": _sample_mesh_surface(gt_mesh, n_points),
        "w_edit": w_edit, "w_stable": 1.0 - w_edit,
        "origin": origin, "spacing": spacing, "resolution": grid_resolution,
    }
    if cache_key is not None:
        if len(REFERENCE_FIELD_CACHE) >= REFERENCE_FIELD_CACHE_LIMIT:
            REFERENCE_FIELD_CACHE.pop(next(iter(REFERENCE_FIELD_CACHE)))
        REFERENCE_FIELD_CACHE[cache_key] = context
    return context


def _trilinear_sample(field, points, origin, spacing, resolution):
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)
    coordinates = np.clip((np.asarray(points, dtype=np.float32) - origin) / spacing, 0, resolution - 1)
    return map_coordinates(field, coordinates.T, order=1, mode="nearest")


def _weighted_one_way_cd(source_points, target_points, source_weights, eps=1e-6):
    if len(source_points) == 0 or len(target_points) == 0:
        raise ValueError("Chamfer distance requires non-empty point clouds")
    weights = np.asarray(source_weights)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Chamfer weights must be finite and nonnegative")
    distances = cKDTree(target_points).query(source_points, k=1)[0]
    return float(np.sum(weights * distances ** 2) / (weights.sum() + eps))


def weighted_chamfer(source_points, target_points, source_weights, target_weights, eps=1e-6):
    return (
        _weighted_one_way_cd(source_points, target_points, source_weights, eps)
        + _weighted_one_way_cd(target_points, source_points, target_weights, eps)
    )


def point_cloud_losses(pred_points, field_context):
    target_points = field_context["gt_points"]
    def weights(points):
        return _trilinear_sample(
            field_context["w_edit"], points, field_context["origin"],
            field_context["spacing"], field_context["resolution"],
        )
    pred_edit, target_edit = weights(pred_points), weights(target_points)
    edit_loss = weighted_chamfer(pred_points, target_points, pred_edit, target_edit)
    # Eq. (17) uses P_target, not P_ori, also for the stable region.
    stable_loss = weighted_chamfer(pred_points, target_points, 1 - pred_edit, 1 - target_edit)
    return edit_loss, stable_loss


def compute_Editcd(pred_mesh, gt_mesh, ori_mesh, n_points=2000, eps=0.01,
                   lambda_local=1.5, lambda_stable=1.0, field_context=None,
                   cache_key=None, grid_resolution=64):
    if field_context is None:
        field_context = _build_reference_field(gt_mesh, ori_mesh, n_points, eps, grid_resolution, cache_key)
    edit_loss, stable_loss = point_cloud_losses(_sample_mesh_surface(pred_mesh, n_points), field_context)
    return lambda_local * edit_loss + lambda_stable * stable_loss, edit_loss, stable_loss


def localized_reward(edit_loss, stable_loss, lambda_edit=1.5, lambda_stable=1.0, invalid_reward=-3.0):
    if edit_loss is None or stable_loss is None or not np.isfinite([edit_loss, stable_loss]).all():
        return float(invalid_reward)
    raw = -lambda_edit * edit_loss - lambda_stable * stable_loss
    return float(np.clip((raw + 1.0) * 10.0, -3.0, 10.0))


# process initializer used in case of forkserver
def init_worker():
    os.environ["OMP_NUM_THREADS"]       = "1"
    os.environ["OPENBLAS_NUM_THREADS"]  = "1"
    os.environ["MKL_NUM_THREADS"]       = "1"
    """
    import trimesh
    from scipy.spatial import cKDTree
    import cadquery as cq

    # make them available to your metric code
    globals()['trimesh'] = trimesh
    globals()['cKDTree'] = cKDTree
    globals()['cq'] = cq
    globals()['compute_normals_metrics'] = compute_normals_metrics"""


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume
        
        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except:
        pass


def normalize_points_like_eval(points):
    scale = np.max(np.abs(points))
    if scale > 1e-12:
        points = points / scale
    return points


def _one_way_squared_nn(source_points, target_points):
    if source_points is None or target_points is None or len(source_points) == 0 or len(target_points) == 0:
        return 0.0

    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=1)
    return float(np.mean(np.square(distances)))


def normalize_point_sets_with_shared_scale(point_sets, reference_point_sets):
    ref_scale = max(
        (np.max(np.abs(points)) for points in reference_point_sets if points is not None and len(points) > 0),
        default=1.0,
    )
    if ref_scale <= 1e-12:
        ref_scale = 1.0
    normalized_sets = []
    for points in point_sets:
        if points is None:
            normalized_sets.append(None)
        else:
            normalized_sets.append(points / ref_scale)
    return normalized_sets, ref_scale


def estimate_surface_tolerance(point_sets, base_eps):
    tolerances = [base_eps]
    for points in point_sets:
        if points is None or len(points) < 2:
            continue
        tree = cKDTree(points)
        neighbor_distances, _ = tree.query(points, k=2)
        nearest_distances = neighbor_distances[:, 1]
        nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
        if nearest_distances.size == 0:
            continue
        # Use the median nearest-neighbor spacing as a lower bound for edit masks
        # so random surface sampling noise does not dominate the reward.
        tolerances.append(float(np.median(nearest_distances)))
    return max(tolerances)


def compute_cd(pred_mesh, gt_mesh, n_points=2000):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    cd = _one_way_squared_nn(pred_points, gt_points) + _one_way_squared_nn(gt_points, pred_points)
    return cd

def get_real_mesh_normalization(reference_mesh):
    if reference_mesh is None or reference_mesh.bounds is None:
        return None, None
    center = (reference_mesh.bounds[0] + reference_mesh.bounds[1]) / 2.0
    extent = np.max(reference_mesh.extents)
    scale = 2.0 / extent if extent > 1e-7 else 1.0
    return center, scale


def transform_real_mesh(mesh, ref_center=None, ref_scale=None):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    if ref_center is None or ref_scale is None:
        ref_center, ref_scale = get_real_mesh_normalization(mesh)
    if ref_center is None or ref_scale is None:
        return mesh
    mesh.apply_translation(-ref_center)
    mesh.apply_scale(ref_scale)
    return mesh


def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)  # shift to center
    extent = np.max(mesh.extents)
    if extent > 1e-7:
            mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def transform_reference_meshes_shared(gt_mesh, ori_mesh):
    """Apply one canonical transform to target and original meshes."""
    bounds = np.concatenate([gt_mesh.bounds, ori_mesh.bounds], axis=0)
    lower = bounds.min(axis=0)
    upper = bounds.max(axis=0)
    center = (lower + upper) / 2.0
    scale = 1.0 / max(float(np.max(upper - lower)), 1e-7)
    for mesh in (gt_mesh, ori_mesh):
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        mesh.apply_translation([0.5, 0.5, 0.5])
    return gt_mesh, ori_mesh


def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)  # Normalize to [0, 1]
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh

def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


class CADRenderingDependencyError(RuntimeError):
    pass


# def code_to_mesh_and_brep_less_safe(code_str):
#     safe_ns = {"cq": cq}
#     ns=safe_ns.copy()
#     #print(f"Executing code {code_str}")
#     try:
#         exec(code_str, ns)
#         mesh = compound_to_mesh(ns["r"].val())
#         # export files if needed
#         # mesh.export(mesh_path)
#         return mesh
#     except Exception as e:
#         print(f"Error executing CadQuery code : {e}")
#         return None
    
def code_to_mesh_and_brep_less_safe(code_str):
    try:
        # Check if the text is empty or just whitespace
        if not code_str or not code_str.strip() or "empty" in code_str:
            return None
        code_str = code_str.strip()
        if code_str.startswith("```"):
            code_str = code_str.split("\n", 1)[1].rsplit("```", 1)[0]
        if "<extrude_end>" not in code_str:
            try:
                import cadquery as cq
            except ImportError as exc:
                raise CADRenderingDependencyError(f"CadQuery is unavailable: {exc}") from exc
            namespace = {"cq": cq, "cadquery": cq, "show_object": lambda *args, **kwargs: None}
            exec(compile(code_str, "<cadquery-completion>", "exec"), namespace)
            result = namespace.get("r")
            if result is None:
                result = namespace.get("result")
            if result is None:
                return None
            shapes = result.vals() if isinstance(result, cq.Workplane) else [result]
            if not shapes or not all(isinstance(shape, cq.Shape) and shape.isValid() for shape in shapes):
                return None
            return compound_to_mesh(cq.Compound.makeCompound(shapes))
            
        # Import the representation's CAD kernel only when it is needed.
        try:
            from utils.parse_seq2obj_rille import CADparser, write_obj_sample
            from utils.visual_obj import OBJReconverter, BRepCheck_Analyzer, write_stl_file
            from utils.geometry.obj_parser import OBJParser
        except ImportError as exc:
            raise CADRenderingDependencyError(f"Command CAD renderer is unavailable: {exc}") from exc
        cad_parser = CADparser(bit=6)
        parsed_data = cad_parser.perform(code_str)
        if parsed_data is None or len(parsed_data) == 0:
            return None
            
        with tempfile.TemporaryDirectory() as tmpdir:
            write_obj_sample(tmpdir, parsed_data)
            
            from utils.visual_obj import find_files
            param_objs = find_files(tmpdir, 'param.obj')
            
            cur_solid = None
            converter = OBJReconverter()
            for obj_path in param_objs:
                parser = OBJParser(obj_path)
                _, faces, meta_info = parser.parse_file(1.0)
                ext_solid, _, _ = converter.parse_obj(faces, meta_info)
                set_op = meta_info["set_op"]
                if set_op == "NewBodyFeatureOperation" or set_op == "JoinFeatureOperation":
                    if cur_solid is None:
                        cur_solid = ext_solid
                    else:
                        cur_solid = converter.my_op(cur_solid, ext_solid, 'fuse')
                elif set_op == "CutFeatureOperation":
                    if cur_solid is not None:
                        cur_solid = converter.my_op(cur_solid, ext_solid, 'cut')
                elif set_op == "IntersectFeatureOperation":
                    if cur_solid is not None:
                        cur_solid = converter.my_op(cur_solid, ext_solid, 'common')
                else:
                    raise Exception("Unknown operation type")
                
                if cur_solid is not None:
                    analyzer = BRepCheck_Analyzer(cur_solid)
                    if not analyzer.IsValid():
                        raise Exception("brep check failed")
            
            if cur_solid is None:
                return None
                
            stl_path = os.path.join(tmpdir, "out.stl")
            write_stl_file(cur_solid, stl_path, linear_deflection=0.001, angular_deflection=0.5)
            
            mesh = trimesh.load_mesh(stl_path)
            return mesh
    except CADRenderingDependencyError:
        raise
    except Exception:
        return None

def get_metrics_from_single_text(text, gt_file, ori_file, n_points, normalize="fixed",
                                grid_resolution=64, tau_phi=0.01, prediction_scale=1.0,
                                mesh_root=None):
    if mesh_root is None and not (Path(gt_file).is_absolute() and Path(ori_file).is_absolute()):
        raise ValueError("Pass mesh_root when reference mesh paths are relative")
    mesh_root = mesh_root or "."
    gt_base_file = os.path.join(mesh_root, gt_file)
    try:
        gt_path, ori_path = Path(gt_base_file), Path(mesh_root, ori_file)
        gt_base_file = str(gt_path) if gt_path.is_file() else str(sorted(gt_path.rglob("*.stl"))[0])
        ori_base_file = str(ori_path) if ori_path.is_file() else str(sorted(ori_path.rglob("*.stl"))[0])
        field_key = (gt_base_file, ori_base_file, n_points, grid_resolution, tau_phi, normalize)
        gt_mesh = trimesh.load_mesh(gt_base_file)
        ori_mesh = trimesh.load_mesh(ori_base_file)
        ref_center, ref_scale = reference_transform(gt_mesh, ori_mesh)
        gt_mesh = apply_reference_transform(gt_mesh, ref_center, ref_scale)
        ori_mesh = apply_reference_transform(ori_mesh, ref_center, ref_scale)
        field_context = _build_reference_field(
            gt_mesh, ori_mesh, n_points, tau_phi, grid_resolution, field_key
        )
    except Exception as e:
        print(f"error preparing LFM reference field for {gt_base_file}: {e}", flush=True)
        return dict(file_name=gt_base_file, cd=None, iou=None, auc=None, edit_cd=None, reward_local=None, reward_stable=None,
                    error_type="reference", error=f"{type(e).__name__}: {e}")

    try:
        pred_mesh = code_to_mesh_and_brep_less_safe(text)
    except Exception as e:
        print(f"Error executing code for {gt_file}: {e}", flush=True)
        return dict(file_name=gt_base_file, cd=None, iou=None, auc=None, edit_cd=None, reward_local=None, reward_stable=None,
                    error_type="dependency", error=f"{type(e).__name__}: {e}")
    if pred_mesh is None:
        #print("Skipping metrics: invalid prediction", flush=True)
        return dict(file_name=gt_base_file, cd=None, iou=None, auc=None, edit_cd=None, reward_local=None, reward_stable=None)
    #t_met = time.perf_counter()
    cd, iou, auc, edit_cd, reward_local, reward_stable = None, None, None, None, None, None
    try:
        pred_mesh = apply_reference_transform(pred_mesh, ref_center, ref_scale, prediction_scale)

        cd = compute_cd(pred_mesh, gt_mesh, n_points)
        edit_cd, reward_local, reward_stable = compute_Editcd(
            pred_mesh, gt_mesh, ori_mesh, n_points, eps=tau_phi,
            lambda_local=1.5, lambda_stable=1.0,
            field_context=field_context, cache_key=field_key,
            grid_resolution=grid_resolution,
        )

        # iou = compute_iou(gt_mesh, pred_mesh)

        #auc, _, _ = compute_normals_metrics(gt_mesh, pred_mesh, n_points=n_points, tol=2)

    except Exception as e:
        print(f"error for {gt_base_file}: {e}", flush=True)
        pass

    #print(f"[TIME] metric computation without cadquery: {time.perf_counter()-t_cad:.3f}s on worker pid={os.getpid()}")
    """"
    finally:
        try:
            if gt_mesh is not None:
                del gt_mesh
            if pred_mesh is not None:
                del pred_mesh
        except:
            pass"""
    return dict(file_name=gt_base_file, cd=cd, iou=iou, auc=auc, edit_cd=edit_cd, reward_local=reward_local, reward_stable=reward_stable)




POOL = None
POOL_SIZE = None
POOL_MAX_TASKS_PER_CHILD = 8

def init_pool(max_workers):
    print("Initializing POOL", flush=True)
    global POOL, POOL_SIZE
    if POOL is None or POOL_SIZE != max_workers:
        close_pool()
        ctx = get_context("spawn")
        POOL = ctx.Pool(
            processes=max_workers,
            initializer=init_worker,
            maxtasksperchild=POOL_MAX_TASKS_PER_CHILD,
        )
        POOL_SIZE = max_workers
    return POOL

def close_pool():
    global POOL, POOL_SIZE
    if POOL is not None:
        POOL.close()
        POOL.join()
        POOL = None
        POOL_SIZE = None


def reset_pool():
    global POOL, POOL_SIZE
    if POOL is not None:
        POOL.terminate()
        POOL.join()
        POOL = None
        POOL_SIZE = None


def get_metrics_from_texts(texts, meshes, originals, max_workers=None, normalize="fixed", timeout=60,
                           grid_resolution=64, tau_phi=0.01, prediction_scale=1.0,
                           mesh_root=None):
    if mesh_root is None and any(not Path(path).is_absolute() for path in (*meshes, *originals)):
        raise ValueError("Pass mesh_root when reference mesh paths are relative")
    worker_count = max_workers if max_workers else 4
    if POOL is None or POOL_SIZE != worker_count:
        init_pool(worker_count)
    # print(f"[POOL] POOL size={POOL._processes} pid={os.getpid()}")
    #t0 = time.perf_counter()
    #print(f"Example of one generated code : {texts[0]}")
    n_points = 2000
    args = [
        (text, gt, ori, n_points, normalize, grid_resolution, tau_phi, prediction_scale, mesh_root)
        for text, gt, ori in zip(texts, meshes, originals)
    ]
    async_results = [POOL.apply_async(get_metrics_from_single_text, args=arg) for arg in args]
    results = []
    should_reset_pool = False
    for res in async_results:
        try:
            output = res.get(timeout=timeout)
        except TimeoutError:
            print("[__TIMEOUT__] metrics task computation ERROR, skipping", flush=True)
            output = None
            should_reset_pool = True
        except Exception as exc:
            print(f"[__CRASH__] metrics task computation ERROR: {exc}", flush=True)
            output = dict(file_name=None, cd=None, iou=None, auc=None, edit_cd=None,
                          reward_local=None, reward_stable=None, error_type="worker",
                          error=f"{type(exc).__name__}: {exc}")
            should_reset_pool = True

        if output is None:
            results.append(dict(file_name=None, cd=None, iou=None, auc=None,edit_cd=None, reward_local=None, reward_stable=None))
        else:
            results.append(output)

    #wait = time.perf_counter() - t0 
    #print(f"TIME to get metrics for {len(texts)} samples : {wait}")
    if should_reset_pool:
        reset_pool()

    return results




def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area under the curve (AUC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    #tol = 0.01 * max(gt_mesh.extents.max(), pred_mesh.extents.max())  # 1% of the mesh extent
    tol = pred_mesh.extents.max() * tol  / 100

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    # normals of sampled points
    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)
    # get the indices of the neighbors for each ground-truth point

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs] # candidates

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)  # (k,)
        best_idx = np.argmax(dots)  # index of the best aligned normal

        valid_pred_normals.append(pn_neighbors[best_idx])  # (3,)

        valid_gt_points.append(gt_points[i])  # (3,)
        valid_pred_points.append(pred_points[idxs[best_idx]])  # (3,)

    if len(valid_pred_normals) == 0:
        return 1.0, 0.0, 100

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100
    #print(f"Number of points with no neighbors within tol: {nb_invalid} out of {n_points} ({per_invalid:.2f}%)")
    
    
    # compute cosine similarity
    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)
    
    # distribution of angles between normals
    angles = np.arccos(cos_sim)
    angles = np.sort(angles)

    # add invalid points to the end of the array with max angle (pi)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N+1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0],   cdf,   [1.0]))
    auc_normalized = trapz(y, x) / np.pi  # Normalize by the maximum possible aoc (which is pi)

    #we want to maximize the AUC
    #aoc_normalized = 1 - auc_normalized
    # plot the aoc
    #if aoc_normalized > 0.3:
        #print(f"HIGH aoc: {aoc_normalized:.2f}")
        #plot_aoc(angles, cdf, title='aoc of Normal Angles', aoc_value=aoc_normalized)

    #print(f"AUC {auc_normalized}", flush=True)
    return auc_normalized, mean_cos_sim, per_invalid
