import os, sys, runpy
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

for mod in ["data", "s1_eda", "s2_reduccion", "s3_clustering", "s4_modelos", "s5_umbrales", "gen_tex"]:
    print(f"\n=== {mod} ===")
    runpy.run_module(mod, run_name="__main__")
print("\nPipeline completo. report/datos.tex regenerado -> recompilar informe con tectonic.")
