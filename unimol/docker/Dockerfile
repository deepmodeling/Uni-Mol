FROM dptechnology/unicore:0.0.1-pytorch1.11.0-cuda11.3

RUN pip install setuptools wheel twine

RUN pip install rdkit-pypi==2021.9.5.1

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    pip cache purge
