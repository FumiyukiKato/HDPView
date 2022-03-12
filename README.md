# HDPView

**HDPView: Differentially Private Materialized View for Exploring High Dimensional Relational Data**


### Souce code Download
```
$ git clone --recursive https://github.com/FumiyukiKato/HDPView.git
```


### Dependencies
- [Docker](https://www.docker.com/)

### Docker Build
```
$ docker build -t dp_ex .
```

### Docker Run
```
$ docker run --name hdpview --user root -v `pwd`:/home/jovyan/work -p 8888:8888 -e GRANT_SUDO=yes -it dp_ex /bin/bash
($ docker start -i hdpview)
```

### Download datasets & Compile dependencies
```
(docker) $ ./compile-dependencies.sh
(docker) $ ./download_dataset.sh
(docker) & python src/script/preprocess.py --dataset all
```

### Experimental script
```
$ (docker) exp/experiment_count_range_query.sh nume-adult 1.0 3
```
