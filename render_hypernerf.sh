python render.py --model_path "output/hypernerf/broom2/"  --skip_train --configs arguments/hypernerf/broom2.py 
python render.py --model_path "output/hypernerf/3dprinter/"  --skip_train  --configs arguments/hypernerf/3dprinter.py
python render.py --model_path "output/hypernerf/chicken/"  --skip_train  --configs arguments/hypernerf/chicken.py
python render.py --model_path "output/hypernerf/peel-banana/"  --skip_train  --configs arguments/hypernerf/banana.py

python metrics.py --model_path "output/hypernerf/broom2/"  
python metrics.py --model_path "output/hypernerf/3dprinter/"  
python metrics.py --model_path "output/hypernerf/chicken/"  
python metrics.py --model_path "output/hypernerf/peel-banana/" 