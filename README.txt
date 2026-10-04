pip install -r requirements.txt
python download_face_weights.py
python train.py --data-root data --device cpu 
python evaluate.py --checkpoint models/baby_classifier.pth --threshold 0.82
python inference.py --baby-threshold 0.82 --conf 0.5