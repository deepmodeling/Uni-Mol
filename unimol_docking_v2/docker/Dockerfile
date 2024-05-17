FROM dptechnology/unicore:latest-pytorch1.12.1-cuda11.6-rdma

RUN pip install rdkit-pypi==2022.9.3
RUN pip install biopandas

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya