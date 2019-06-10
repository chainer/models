#!/bin/bash
# VOT
git clone https://github.com/yuyu2172/trackdat.git
cd trackdat
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
bash scripts/unpack_vot.sh dl/vot2018 ../VOT2018
cp dl/vot2018/list.txt ../VOT2018/
cd .. && rm -rf ./trackdat

# json file for eval toolkit
# wget http://www.robots.ox.ac.uk/~qwang/VOT2018.json
python create_json.py VOT2018

# DAVIS
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
ln -s ./DAVIS ./DAVIS2016
ln -s ./DAVIS ./DAVIS2017
