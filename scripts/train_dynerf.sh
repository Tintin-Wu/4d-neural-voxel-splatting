exp_name=dynerf
python train.py -s data/dynerf/flame_salmon_1 --port 6468 --expname "$exp_name/flame_salmon_1" --configs arguments/$exp_name/flame_salmon_1.py 
python train.py -s data/dynerf/coffee_martini --port 6472 --expname "$exp_name/coffee_martini" --configs arguments/$exp_name/coffee_martini.py  
# wait
python train.py -s data/dynerf/cook_spinach --port 6436 --expname "$exp_name/cook_spinach" --configs arguments/$exp_name/cook_spinach.py 
python train.py -s data/dynerf/cut_roasted_beef --port 6470 --expname "$exp_name/cut_roasted_beef" --configs arguments/$exp_name/cut_roasted_beef.py 
# wait 
python train.py -s data/dynerf/flame_steak      --port 6471 --expname "$exp_name/flame_steak" --configs arguments/$exp_name/flame_steak.py 
# Need re run
python train.py -s data/dynerf/sear_steak       --port 6569 --expname "$exp_name/sear_steak" --configs arguments/$exp_name/sear_steak.py  
# wait

python render.py --model_path output/$exp_name/cut_roasted_beef --configs arguments/$exp_name/cut_roasted_beef.py --skip_train 
python render.py --model_path output/$exp_name/sear_steak --configs arguments/$exp_name/sear_steak.py --skip_train 
# wait
python render.py --model_path output/$exp_name/flame_steak --configs arguments/$exp_name/flame_steak.py --skip_train 
python render.py --model_path output/$exp_name/flame_salmon_1 --configs arguments/$exp_name/flame_salmon_1.py --skip_train 
# wait
python render.py --model_path output/$exp_name/cook_spinach  --configs arguments/$exp_name/cook_spinach.py --skip_train  
python render.py --model_path output/$exp_name/coffee_martini --configs arguments/$exp_name/coffee_martini.py --skip_train 
# wait
python metrics.py --model_path "output/$exp_name/cut_roasted_beef/"  
python metrics.py --model_path "output/$exp_name/cook_spinach/" 
# wait
python metrics.py --model_path "output/$exp_name/sear_steak/" 
python metrics.py --model_path "output/$exp_name/flame_salmon_1/"  
# wait
python metrics.py --model_path "output/$exp_name/flame_steak/" 
python metrics.py --model_path "output/$exp_name/coffee_martini/" 
echo "Done"