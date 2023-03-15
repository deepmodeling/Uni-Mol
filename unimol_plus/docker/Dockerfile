FROM dptechnology/unicore:latest-pytorch1.12.1-cuda11.6-rdma

RUN pip install rdkit==2022.09.3

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya
