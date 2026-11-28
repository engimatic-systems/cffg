FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b
COPY packages/numpy /usr/local/lib/python3.12/site-packages/numpy
COPY packages/numpy.libs /usr/local/lib/python3.12/site-packages/numpy.libs
COPY packages/numpy-2.5.1.dist-info /usr/local/lib/python3.12/site-packages/numpy-2.5.1.dist-info
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
CMD ["python", "-IB", "-"]
