python -B test_cls/inference_cls_2.py \
    --model-folder results_classification_cv_updated/h-optimus_attention_20250914_231455 \
    --test-csv data/manifests/test_features_h-optimus.csv \
    --run-interpretability \
    --h5-dir data/raw/BCR_NET \
    --tfrecord-dir data/raw/UCMC \
    --visualize-patches \
    --n-patches-per-category 5 \
    --n-visualize 10