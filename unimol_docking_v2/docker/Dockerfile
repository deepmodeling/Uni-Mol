FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

RUN pip install rdkit-pypi==2022.9.3
RUN pip install biopandas

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    conda clean -ya