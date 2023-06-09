download-dataset:
    mkdir -p data
    curl --continue-at - http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip --output data/DeepStab.zip
    unzip data/DeepStab.zip -d data
