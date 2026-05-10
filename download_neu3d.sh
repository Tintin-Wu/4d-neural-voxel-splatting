unzip ../zip_files/coffee_martini.zip -d ../data/dynerf/
unzip ../zip_files/cook_spinach.zip -d ../data/dynerf/
unzip ../zipfiles/cut_roasted_beef.zip -d ../data/dynerf/
zip -F ../zip_files/flame_salmon_1_split.zip --out ../zip_files/flame_salmon_1.zip
unzip ../zip_files/flame_salmon_1.zip -d ../data/dynerf/
unzip ../zip_files/flame_steak.zip -d ../data/dynerf/
unzip ../zip_files/sear_steak.zip -d ../data/dynerf/

# bash ./scripts/preprocess_dynerf.sh
# wait

# bash colmap.sh data/dynerf/coffee_martini llff
# bash colmap.sh data/dynerf/cook_spinach llff
# bash colmap.sh data/dynerf/cut_roasted_beef llff
# bash colmap.sh data/dynerf/flame_salmon_1 llff
# bash colmap.sh data/dynerf/flame_steak llff
# bash colmap.sh data/dynerf/sear_steak llff
