FROM nvcr.io/nvidia/pytorch:19.12-py3

WORKDIR /DistSup

RUN apt-get update && apt-get install -y p7zip-full rsync

COPY environment.yml environment.yml

RUN conda env create -f environment.yml

RUN conda init bash && \
    echo "conda activate distsup" >> ~/.bashrc ; \
    echo "export DISTSUP_DIR=/DistSup" >> ~/.bashrc ; \
    echo "export PYTHONPATH=$PYTHONPATH:/DistSup" >> ~/.bashrc

COPY . .

CMD [ "/bin/bash" ]
