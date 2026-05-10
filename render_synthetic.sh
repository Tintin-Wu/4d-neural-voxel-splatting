python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
python render.py --model_path "output/dnerf/hellwarrior/"  --skip_train  --configs arguments/dnerf/hellwarrior.py
python render.py --model_path "output/dnerf/hook/"  --skip_train  --configs arguments/dnerf/hook.py
python render.py --model_path "output/dnerf/jumpingjacks/"  --skip_train  --configs arguments/dnerf/jumpingjacks.py
python render.py --model_path "output/dnerf/lego/"  --skip_train  --configs arguments/dnerf/lego.py
python render.py --model_path "output/dnerf/mutant/"  --skip_train  --configs arguments/dnerf/mutant.py
python render.py --model_path "output/dnerf/standup/"  --skip_train --configs arguments/dnerf/standup.py
python render.py --model_path "output/dnerf/trex/"  --skip_train  --configs arguments/dnerf/trex.py

python metrics.py --model_path "output/dnerf/bouncingballs/" 
python metrics.py --model_path "output/dnerf/hellwarrior/"  
python metrics.py --model_path "output/dnerf/hook/"  
python metrics.py --model_path "output/dnerf/jumpingjacks/"  
python metrics.py --model_path "output/dnerf/lego/"  
python metrics.py --model_path "output/dnerf/mutant/"
python metrics.py --model_path "output/dnerf/standup/" 
python metrics.py --model_path "output/dnerf/trex/"
