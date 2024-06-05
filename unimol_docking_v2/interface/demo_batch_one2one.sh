python demo.py --mode batch_one2one --batch-size 8 --conf-size 10 --cluster \
        --input-batch-file input_batch_one2one.csv \
        --output-ligand-dir predict_sdf \
        --steric-clash-fix \
        --model-dir checkpoint_best.pt
