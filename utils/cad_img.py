import argparse
from OCC.Core.Graphic3d import *
from OCC.Display.OCCViewer import Viewer3d
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_WHITE
from OCC.Core.V3d import V3d_DirectionalLight
from OCC.Core.gp import gp_Dir
from glob import glob
import pathlib
from tqdm import tqdm
import os
import traceback
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
# os.environ["GALLIUM_DRIVER"] = "llvmpipe"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["DISABLE_X11"] = "1"
# os.environ["QT_QPA_PLATFORM"] = "offscreen"
def render(shape, filename, width=1024, height=768, face_color_rgb=(0.2, 0.2, 0.2), edge_color_rgb=(0, 0, 0), show_face_boundary=True):
    viewer = None
    try:
        viewer = Viewer3d()
        viewer.Create(phong_shading=True, create_default_lights=True)
        viewer.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
        viewer.SetModeShaded()
        viewer.hide_triedron()
        viewer.EnableAntiAliasing()
        dir_light = V3d_DirectionalLight(gp_Dir(0, 0.5, -1), Quantity_Color(Quantity_NOC_WHITE))
        dir_light.SetEnabled(True)
        dir_light.SetIntensity(500.0)
        viewer.Viewer.AddLight(dir_light)
        viewer.Viewer.SetLightOn()

        viewer.default_drawer.EnableDrawHiddenLine()
        viewer.default_drawer.SetFaceBoundaryDraw(show_face_boundary)
        ais_context = viewer.GetContext()
        dc = ais_context.DeviationCoefficient()
        da = ais_context.DeviationAngle()
        factor = 10
        ais_context.SetDeviationCoefficient(dc / factor)
        ais_context.SetDeviationAngle(da / factor)
        topexp = TopologyExplorer(shape)
        for face in topexp.faces():
            if face is not None:
                viewer.DisplayShape(face, color=Quantity_Color(*face_color_rgb, Quantity_TOC_RGB))
        for edge in topexp.edges():
            if edge is not None:
                viewer.DisplayShape(edge, color=Quantity_Color(*edge_color_rgb, Quantity_TOC_RGB))
        viewer.FitAll()
        viewer.SetSize(width, height)
        viewer.View.Dump(str(filename))
    finally:
        # OCC keeps native viewer resources alive unless explicitly released.
        if viewer is not None:
            try:
                viewer.EraseAll()
            except Exception:
                pass
            try:
                viewer.Finalize()
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Input folder of STP/STEP files")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder of PNG files")
    p.add_argument("--width", type=int, default=1024, help="Width of image")
    p.add_argument("--height", type=int, default=768, help="Height of image")
    p.add_argument("--error_log", type=str, default=None, help="Path for failed-file diagnostics")
    p.add_argument("--overwrite", action="store_true", help="Re-render existing PNG files")

    args = p.parse_args()
    
    input_root = pathlib.Path(args.input_dir)
    files = sorted(set(input_root.glob("*/*.step")) | set(input_root.glob("*/*.stp")))
    print(f"found STEP files: {len(files)}")
    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    error_log = pathlib.Path(args.error_log) if args.error_log else output_path / "render_errors.log"
    success_count = 0
    failure_count = 0
    skipped_count = 0
    with error_log.open("w", encoding="utf-8") as errors:
        for fn in tqdm(files, desc="Rendering STEP files"):
            output_file = output_path / f"{fn.stem}.png"
            if output_file.exists() and not args.overwrite:
                skipped_count += 1
                continue
            try:
                shape = read_step_file(str(fn))
                if shape is None or shape.IsNull():
                    raise RuntimeError("read_step_file returned a null shape")
                render(shape, output_path.joinpath(fn.stem + ".png"), args.width, args.height)
                success_count += 1
            except Exception as exc:
                failure_count += 1
                errors.write(f"{fn}\n{type(exc).__name__}: {exc}\n")
                errors.write(traceback.format_exc())
                errors.write("\n")
                errors.flush()
                print(f"[ERROR] {fn}: {type(exc).__name__}: {exc}", flush=True)

    print(f"success: {success_count}")
    print(f"failed: {failure_count}")
    print(f"skipped existing: {skipped_count}")
    print(f"error log: {error_log}")

if __name__ == "__main__":
    main()
