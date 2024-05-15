python demo.py --input-protein ../example_data/protein.pdb \
        --input-ligand ../example_data/ligand.sdf \
        --input-docking-grid ../example_data/docking_grid.json \
        --output-ligand-name ligand_predict \
        --output-ligand-dir predict_sdf \
        --model-dir checkpoint_best.pt