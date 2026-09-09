# Runs the three.js demos (demos/server.py) on a Hugging Face Docker Space.
# CPU-only torch keeps the image small; the server needs nothing else.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "numpy>=1.24"

# Spaces run the container as a non-root user with uid 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user spinor_lib ./spinor_lib
COPY --chown=user demos ./demos

EXPOSE 7860
CMD ["python", "demos/server.py", "--host", "0.0.0.0", "--port", "7860"]
