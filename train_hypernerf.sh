exp_name='hypernerf'
python train.py -s  ../data/hypernerf/vrig/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py
python train.py -s  ../data/hypernerf/vrig/3dprinter/ --port 6017 --expname "hypernerf/3dprinter" --configs arguments/hypernerf/3dprinter.py
python train.py -s  ../data/hypernerf/vrig/chicken/ --port 6017 --expname "hypernerf/chicken" --configs arguments/hypernerf/chicken.py
python train.py -s  ../data/hypernerf/vrig/peel-banana/ --port 6017 --expname "hypernerf/peel-banana" --configs arguments/hypernerf/banana.py
python train.py -s ../data/hypernerf/aleks-teapot --port 6568 --expname "$exp_name/aleks-teapot" --configs arguments/$exp_name/default.py 
python train.py -s ../data/hypernerf/slice-banana --port 6566 --expname "$exp_name/slice-banana" --configs arguments/$exp_name/default.py 
python train.py -s ../data/hypernerf/chickchicken --port 6569 --expname "$exp_name/interp-chicken" --configs arguments/$exp_name/default.py
python train.py -s ../data/hypernerf/cut-lemon1 --port 6670 --expname $exp_name/cut-lemon1 --configs arguments/$exp_name/default.py
python train.py -s ../data/hypernerf/hand1-dense-v2 --port 6671 --expname $exp_name/hand1-dense-v2 --configs arguments/$exp_name/default.py
python train.py -s ../data/hypernerf/torchocolate --port 6672 --expname $exp_name/torchocolate --configs arguments/$exp_name/default.py

wait

python render.py --model_path "output/hypernerf/broom2/"  --skip_train --configs arguments/hypernerf/broom2.py 
python render.py --model_path "output/hypernerf/3dprinter/"  --skip_train  --configs arguments/hypernerf/3dprinter.py
python render.py --model_path "output/hypernerf/chicken/"  --skip_train  --configs arguments/hypernerf/chicken.py
python render.py --model_path "output/hypernerf/peel-banana/"  --skip_train  --configs arguments/hypernerf/banana.py
python render.py --model_path "output/$exp_name/aleks-teapot" --configs arguments/$exp_name/default.py --skip_train 
python render.py --model_path "output/$exp_name/slice-banana"  --configs arguments/$exp_name/default.py --skip_train 
python render.py --model_path "output/$exp_name/interp-chicken" --configs arguments/$exp_name/default.py --skip_train 
python render.py --model_path "output/$exp_name/cut-lemon1"  --configs arguments/$exp_name/default.py --skip_train 
python render.py --model_path "output/$exp_name/hand1-dense-v2"  --configs arguments/$exp_name/default.py --skip_train
python render.py --model_path "output/$exp_name/torchocolate" --configs arguments/$exp_name/default.py --skip_train 

wait

python metrics.py --model_path "output/hypernerf/broom2/"  
python metrics.py --model_path "output/hypernerf/3dprinter/"  
python metrics.py --model_path "output/hypernerf/chicken/"  
python metrics.py --model_path "output/hypernerf/peel-banana/" 
python metrics.py --model_path "output/hypernerf/aleks-teapot/"  
python metrics.py --model_path "output/$exp_name/slice-banana/" 
python metrics.py --model_path "output/$exp_name/interp-chicken/" 
python metrics.py --model_path "output/$exp_name/cut-lemon1/" 
python metrics.py --model_path "output/$exp_name/hand1-dense-v2/" 
python metrics.py --model_path "output/$exp_name/torchocolate/" 

echo "Done"