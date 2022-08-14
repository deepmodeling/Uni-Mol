FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

RUN conda install -y -c conda-forge rdkit==2021.09.5 && conda clean -ya

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya
